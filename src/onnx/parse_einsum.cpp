/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/permutation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_einsum : op_parser<parse_einsum>
{
    using int_mat = std::vector<std::vector<int>>;

    struct EquationInfo
    {
        std::vector<std::string> input_terms;
        std::string output_term;
        std::map<char, int> label_count;
        bool explicit_form;
        std::string unique_labels;
        size_t ellipses_ndim;
    };

    std::vector<op_desc> operators() const { return {{"Einsum"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(not contains(info.attributes, "equation"))
            MIGRAPHX_THROW("Equation attribute is required");
        std::string equation = info.attributes.at("equation").s();

        const EquationInfo eq_info = analyze_equation(equation, args);
        auto terms                 = eq_info.input_terms;
        terms.push_back(eq_info.output_term);

        const auto map_mat =
            make_mapping_matrix(terms, eq_info.unique_labels, eq_info.ellipses_ndim);
        const auto duplicates = find_duplicates(terms);

        // Holds the mapping matrix representations of the two terms being processed
        // cur_pair[0] acts as the accumulator for previously processed inputs
        // cur_pair[1] holds the representation for the current input
        // As operations are added to the einsum graph, cur_pair gets manipulated
        int_mat cur_pair = make_matrix(2, map_mat[0].size(), -1);

        instruction_ref cur_op;
        std::optional<instruction_ref> last_op;
        // Perform a left fold on the inputs
        for(auto arg_idx = 0; arg_idx < args.size(); ++arg_idx)
        {
            cur_op      = args[arg_idx];
            cur_pair[1] = map_mat[arg_idx];

            cur_op =
                preprocess_input(info, cur_op, duplicates[arg_idx], map_mat, arg_idx, cur_pair);

            if(last_op)
                cur_op = process_pair(info, *last_op, cur_op, map_mat, arg_idx, cur_pair);

            last_op     = cur_op;
            cur_pair[0] = cur_pair[1];
        }

        return finalize_output(info, cur_op, map_mat, cur_pair);
    }

    // Equation Parsing

    EquationInfo analyze_equation(std::string_view equation,
                                  const std::vector<instruction_ref>& args) const
    {
        EquationInfo eq_info = parse_equation(equation);

        eq_info.ellipses_ndim = validate_input_terms(eq_info.input_terms, args);
        if(not eq_info.output_term.empty())
            validate_output_term(eq_info.output_term, eq_info.label_count, eq_info.ellipses_ndim);
        else if(not eq_info.explicit_form)
            eq_info.output_term = generate_output_term(eq_info.label_count, eq_info.ellipses_ndim);

        for(const auto [l, _] : eq_info.label_count)
            eq_info.unique_labels += l;

        return eq_info;
    }

    EquationInfo parse_equation(std::string_view equation) const
    {
        EquationInfo ret;

        std::string term;
        bool has_ellipsis = false;
        ret.explicit_form = false;

        for(int i = 0; i < equation.size(); ++i)
        {
            const char c = equation[i];
            switch(c)
            {
            case ' ': break;
            case '-':
                if(ret.explicit_form)
                    MIGRAPHX_THROW("Einsum equation has multiple '->' symbols");

                if(i + 1 >= equation.size() or equation[i + 1] != '>')
                    MIGRAPHX_THROW("Invalid '->' in einsum equation");

                ++i;
                ret.explicit_form = true;
                [[fallthrough]];
            case ',':
                has_ellipsis = false;
                ret.input_terms.emplace_back(term);
                term.clear();
                break;
            case '.':
                if(has_ellipsis)
                    MIGRAPHX_THROW("Ellipsis can only appear once per einsum equation term");

                if(i + 2 >= equation.size() or equation[i + 1] != '.' or equation[i + 2] != '.')
                    MIGRAPHX_THROW("Incomplete ellipsis in einsum equation " +
                                   std::string(equation));

                i += 2;
                has_ellipsis = true;
                term += '*';
                break;
            default:
                if(std::isalpha(c) == 0)
                    MIGRAPHX_THROW(std::string("Invalid character '") + c +
                                   "' in einsum equation term");

                term += c;
                if(not ret.explicit_form)
                    ++ret.label_count[c];
            }
        }

        if(ret.explicit_form)
            ret.output_term = term;
        else
            ret.input_terms.push_back(term);

        return ret;
    }

    size_t validate_input_terms(const std::vector<std::string>& input_terms,
                                const std::vector<instruction_ref>& args) const
    {
        if(input_terms.size() != args.size())
            MIGRAPHX_THROW("Number of terms in the input equation - " +
                           std::to_string(input_terms.size()) +
                           " does not match the number of inputs " + std::to_string(args.size()));

        auto global_ellipses_dims = 0u;
        for(auto i = 0u; i < args.size(); ++i)
        {
            const auto& term = input_terms[i];
            const auto dims  = args[i]->get_shape().lens();
            const auto rank  = dims.size();

            auto current_dim = 0u;
            for(const auto l : term)
            {
                if(l == '*')
                {
                    const auto ellipses_dims = rank - term.size() + 1;
                    if(global_ellipses_dims > 0 and ellipses_dims != global_ellipses_dims)
                        MIGRAPHX_THROW("Every occurrence of ellipsis in the equation must "
                                       "represent the same number of dimensions");
                    global_ellipses_dims = ellipses_dims;
                    current_dim += ellipses_dims;
                }
                else
                    ++current_dim;
            }

            if(current_dim != rank)
                MIGRAPHX_THROW("Number of labels in " + std::to_string(i + 1) + ". input_term (" +
                               term + ") does not match the rank (" + std::to_string(rank) +
                               ") of corresponding input");
        }

        return global_ellipses_dims;
    }

    void validate_output_term(std::string_view output_term,
                              const std::map<char, int>& label_count,
                              size_t ellipses_ndim) const
    {
        const auto* it = std::find_if(output_term.begin(), output_term.end(), [&](auto l) {
            return not contains(label_count, l) and l != '*';
        });
        if(it != output_term.end())
            MIGRAPHX_THROW("Output term contains label " + std::to_string(*it) +
                           ", which is not present in any of the input terms");

        if(ellipses_ndim != 0 and not contains(output_term, "*"))
            MIGRAPHX_THROW(
                "Output term does not contain ellipsis (...) even though an input term does");
    }

    // Creates output term when the equation is in implicit mode.
    // The created output term must contain the alphabetically sorted sequence of labels appearing
    // exactly once in the equation.
    // If ellipsis are present in the left hand side of the equation, the ellipsis dimensions are
    // set to the beginning of the output term.
    std::string generate_output_term(const std::map<char, int>& label_count,
                                     size_t ellipsis_ndim) const
    {
        std::string output_term = ellipsis_ndim == 0 ? "" : "*";
        for(const auto [label, count] : label_count)
            if(count == 1)
                output_term += label;

        return output_term;
    }

    // Creates a matrix representation of the equation.
    //
    // Rows correspond to equation terms, in order of appearance.
    //
    // Columns represent the unique labels contained in the equation, ordered alphabetically. If
    // ellipses are present in the equation, they are represented by the final N columns(N being the
    // number of dimensions covered by and ellipsis).
    // Labels not present in a given term are signified by -1.
    // Labels present in a given term are signified by the input axis they represent.
    //
    // e.g. For equation "...ik,kj...->ij...", assuming ... cover two dimensions, the resulting
    // matrix is:
    // +-------+----+----+----+---+---+
    // |       | i  | j  | k  | * | * |
    // +-------+----+----+----+---+---+
    // | ...ik |  2 | -1 |  3 | 0 | 1 |
    // | kj... | -1 |  1 |  0 | 2 | 3 |
    // | ij... |  0 |  1 | -1 | 2 | 3 |
    // +-------+----+----+----+---+---+
    int_mat make_mapping_matrix(const std::vector<std::string>& terms,
                                std::string_view unique_labels,
                                size_t ellipses_ndim) const
    {
        std::map<char, int> label_to_column;
        for(auto i = 0; i < unique_labels.size(); ++i)
            label_to_column[unique_labels[i]] = i;

        int_mat map_mat = make_matrix(terms.size(), unique_labels.size() + ellipses_ndim, -1);

        for(auto i = 0; i < terms.size(); ++i)
        {
            const auto& term = terms[i];
            int col_id       = 0;
            for(const auto l : term)
            {
                if(l == '*')
                {
                    std::iota(map_mat[i].end() - ellipses_ndim, map_mat[i].end(), col_id);
                    col_id += ellipses_ndim;
                }
                else
                    map_mat[i][label_to_column[l]] = col_id++;
            }
        }

        return map_mat;
    }

    // Finds the duplicated labels in each of the terms and stores the axes on which they occur.
    //
    // e.g. For equation "iikjj,jkj", the result is a vector containing the two following maps:
    // result[0]: {'i': [0, 1], 'j': [3, 4]}
    // result[1]: {'j': [0, 2]}
    std::vector<std::map<char, std::vector<int>>>
    find_duplicates(const std::vector<std::string>& terms) const
    {
        std::vector<std::map<char, std::vector<int>>> duplicates;
        for(const auto& term : terms)
        {
            std::map<char, std::vector<int>> duplicate_axes;
            for(auto i = 0; i < term.size(); ++i)
                duplicate_axes[term[i]].push_back(i);

            erase_if(duplicate_axes, [](const auto& p) { return p.second.size() < 2; });
            duplicates.push_back(duplicate_axes);
        }

        return duplicates;
    }

    // Graph Building

    instruction_ref preprocess_input(const onnx_parser::node_info& info,
                                     instruction_ref op,
                                     const std::map<char, std::vector<int>>& duplicates,
                                     const int_mat& map_mat,
                                     size_t input_idx,
                                     int_mat& cur_pair) const
    {
        if(not duplicates.empty())
        {
            std::vector<std::vector<int>> diag;
            for(const auto& [_, v] : duplicates)
                diag.push_back(v);

            op = gather_diagonal(info, cur_pair, op, diag);
        }

        // Unsqueeze the input shape in the dimensions marked as -1 in the mapping_matrix
        // Transpose the input shape so the labels are in alphabetical order
        op = unsqueeze_transpose(info, cur_pair, op);

        std::vector<int> red;
        // Check if a given label appears in any of the subsequent mapping matrix terms(this
        // includes the output). If does not, it is reduced and marked as -1 in cur_pair.
        for(int d = 0; d < map_mat[0].size(); ++d)
        {
            bool all_neg_one = all_of(extract_column(map_mat, d, input_idx + 1, map_mat.size()),
                                      [](auto i) { return i == -1; });
            if(all_neg_one and cur_pair[1][d] != -1 and cur_pair[0][d] == -1)
                red.push_back(d);
        }

        return apply_reduce_sum_op(info, op, red, cur_pair[1]);
    }

    instruction_ref gather_diagonal(const onnx_parser::node_info& info,
                                    int_mat& cur_pair,
                                    instruction_ref op,
                                    const int_mat& diag) const
    {
        if(diag.size() != 1)
            MIGRAPHX_THROW(
                "Parsing of equations with more than one duplicated labels per input term is not "
                "implemented");

        const auto& op_lens = op->get_shape().lens();

        int first_axis               = diag[0][0];
        const std::vector<int>& axes = diag[0];
        if(not all_of(axes, [&](int a) { return op_lens[first_axis] == op_lens[a]; }))
            MIGRAPHX_THROW("All duplicate labels have to be the same dimension");

        std::vector<int> batch_axes = set_difference(arange(0, op_lens.size()), axes);
        if(not all_of(batch_axes, [&](int ba) { return ba < axes.front(); }))
            MIGRAPHX_THROW(
                "Parsing of equations with duplicated labels and batch axes that are not "
                "the outer-most axes, is not implemented");

        size_t batch_size = calc_dim(batch_axes, op_lens);

        std::vector<size_t> indices;
        for(size_t batch = 0; batch < batch_size; ++batch)
        {
            for(size_t i = 0; i < op_lens[first_axis]; ++i)
            {
                std::vector<size_t> index(axes.size(), i);
                indices.insert(indices.end(), index.begin(), index.end());
            }
        }

        std::vector<size_t> indices_lens{op_lens[first_axis], axes.size()};
        if(batch_size > 1)
            indices_lens.insert(indices_lens.begin(), batch_size);

        auto indices_arg = info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, indices_lens}, indices});

        op = info.add_instruction(
            migraphx::make_op("gathernd", {{"batch_dims", batch_axes.size()}}), op, indices_arg);

        // compute output row
        for(auto& r : cur_pair[1])
            if(contains(axes, r))
                r = first_axis;

        std::vector<int> to_remove(axes.begin() + 1, axes.end());
        for(auto t : to_remove)
        {
            std::transform(cur_pair[1].begin(),
                           cur_pair[1].end(),
                           cur_pair[1].begin(),
                           [t](auto r) { return r > t ? r - 1 : r; });
        }

        return op;
    }

    instruction_ref process_pair(const onnx_parser::node_info& info,
                                 instruction_ref op1,
                                 instruction_ref op2,
                                 const int_mat& map_mat,
                                 size_t input_idx,
                                 int_mat& cur_pair) const
    {
        // Label is present in current two terms and somewhere in subsequent terms
        std::vector<int> batch_axes;
        // Label is present in only left term
        std::vector<int> left_only;
        // Label is present in only right term
        std::vector<int> right_only;
        // Label is present in current two terms, but not in the subsequent terms
        std::vector<int> sum_axes;

        auto not_neg_one = [](auto i) { return i != -1; };
        // Categorize axes according to label distribution in equation
        for(int d = 0; d < map_mat[0].size(); ++d)
        {
            // The label is present in both terms of cur_pair
            if(all_of(extract_column(cur_pair, d, 0, cur_pair.size()), not_neg_one))
            {
                // The label is present in at least one of the subsequent terms
                if(any_of(extract_column(map_mat, d, input_idx + 1, map_mat.size()), not_neg_one))
                    batch_axes.push_back(d);
                else
                    sum_axes.push_back(d);
            }
            // The label is missing in one or both of the cur_pair
            else
            {
                if(cur_pair[0][d] >= 0)
                    left_only.push_back(d);
                else if(cur_pair[1][d] >= 0)
                    right_only.push_back(d);
                else
                    batch_axes.push_back(d);
            }
        }

        // Permute the inputs so batch_axes are outermost axes and sum_axes are innermost axes
        auto perm = concat_vectors(batch_axes, left_only, right_only, sum_axes);
        op1       = apply_transpose_op(info, op1, perm, cur_pair[0]);
        op2       = apply_transpose_op(info, op2, perm, cur_pair[1]);

        auto new_batch_axes = arange(0, batch_axes.size());
        auto new_sum_axes   = arange(perm.size() - sum_axes.size(), perm.size());

        auto common_labels = set_union(new_batch_axes, new_sum_axes);
        std::tie(op1, op2) = apply_broadcast_op(info, op1, op2, common_labels);

        auto op = batch_dot(info, cur_pair, op1, op2, new_batch_axes, new_sum_axes);

        auto perm_cpy = perm;
        for(auto i = 0; i < perm.size(); ++i)
            perm[perm_cpy[i]] = i;

        return apply_transpose_op(info, op, perm, cur_pair[1]);
    }

    instruction_ref batch_dot(const onnx_parser::node_info& info,
                              int_mat& cur_pair,
                              instruction_ref op1,
                              instruction_ref op2,
                              const std::vector<int>& batch_axes,
                              const std::vector<int>& sum_axes) const
    {
        auto op1_lens = op1->get_shape().lens();
        auto op2_lens = op2->get_shape().lens();

        std::vector<ssize_t> dims1{static_cast<ssize_t>(calc_dim(batch_axes, op1_lens)),
                                   -1,
                                   static_cast<ssize_t>(calc_dim(sum_axes, op1_lens))};
        std::vector<ssize_t> dims2{static_cast<ssize_t>(calc_dim(batch_axes, op2_lens)),
                                   -1,
                                   static_cast<ssize_t>(calc_dim(sum_axes, op2_lens))};

        op1 = info.add_instruction(make_op("reshape", {{"dims", dims1}}), op1);
        op2 = info.add_instruction(make_op("reshape", {{"dims", dims2}}), op2);
        op2 = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), op2);
        instruction_ref op = info.add_instruction(make_op("dot"), op1, op2);

        std::vector<size_t> new_lens(op1_lens.size(), 1);
        std::transform(op1_lens.begin(),
                       op1_lens.begin() + (new_lens.size() - sum_axes.size()),
                       op2_lens.begin(),
                       new_lens.begin(),
                       [](auto len1, auto len2) { return std::max(len1, len2); });

        op = info.add_instruction(make_op("reshape", {{"dims", new_lens}}), op);

        // compute output row
        std::transform(cur_pair[0].begin(),
                       cur_pair[0].end(),
                       cur_pair[1].begin(),
                       cur_pair[1].begin(),
                       [](int lhs, int rhs) { return std::max(lhs, rhs); });
        for(int a : sum_axes)
            cur_pair[1][a] = -1;

        return op;
    }

    instruction_ref finalize_output(const onnx_parser::node_info& info,
                                    instruction_ref op,
                                    const int_mat& map_mat,
                                    int_mat& cur_pair) const
    {
        if(any_of(map_mat.back(), [](auto i) { return i >= 0; }))
        {
            cur_pair[1] = map_mat.back();
            std::vector<int> red;
            for(int d = 0; d < map_mat[0].size(); ++d)
            {
                if(cur_pair[0][d] > 0 and cur_pair[1][d] == -1)
                    red.push_back(d);
            }

            op = apply_reduce_sum_op(info, op, red, cur_pair[1]);
        }

        return transpose_squeeze(info, cur_pair, op, map_mat.back());
    }

    instruction_ref unsqueeze_transpose(const onnx_parser::node_info& info,
                                        int_mat& cur_pair,
                                        instruction_ref op) const
    {
        std::vector<int> unsq_axes;
        std::vector<std::tuple<int, int>> perm;

        for(auto i = 0; i < cur_pair[1].size(); ++i)
        {
            if(cur_pair[1][i] == -1)
                unsq_axes.push_back(i);
            else
                perm.push_back({cur_pair[1][i], i});
        }
        auto unsqueeze = info.add_instruction(make_op("unsqueeze", {{"axes", unsq_axes}}), op);

        std::sort(
            perm.begin(), perm.end(), by(std::less<>{}, [](auto x) { return std::get<0>(x); }));

        std::vector<int> new_perm(cur_pair[1].size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        for(auto i = 0, p = 0; i < cur_pair[1].size(); ++i)
        {
            if(cur_pair[1][i] == -1)
                continue;

            new_perm[std::get<1>(perm[p++])] = i;
        }

        return apply_transpose_op(info, unsqueeze, new_perm, cur_pair[1]);
    }

    instruction_ref transpose_squeeze(const onnx_parser::node_info& info,
                                      int_mat& cur_pair,
                                      instruction_ref op,
                                      std::vector<int> row_output) const
    {
        std::vector<std::tuple<int, int>> perm;
        std::vector<int> sq;

        for(auto i = 0; i < row_output.size(); ++i)
        {
            if(row_output[i] == -1)
                sq.push_back(i);
            else
                perm.push_back({row_output[i], i});
        }

        std::sort(
            perm.begin(), perm.end(), by(std::less<>{}, [](auto x) { return std::get<0>(x); }));

        std::vector<int> new_perm(cur_pair[1].size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        for(auto i = 0, p = 0; i < row_output.size(); ++i)
        {
            if(row_output[i] == -1)
                continue;

            new_perm[i] = std::get<1>(perm[p++]);
        }

        op = apply_transpose_op(info, op, new_perm, cur_pair[1]);

        if(not sq.empty())
        {
            op = info.add_instruction(make_op("squeeze", {{"axes", sq}}), op);
            // compute output row
            for(int a : sq)
                cur_pair[1][a] = -1;
        }

        return op;
    }

    instruction_ref apply_transpose_op(const onnx_parser::node_info& info,
                                       instruction_ref op,
                                       const std::vector<int>& perm,
                                       std::vector<int>& row) const
    {
        op = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op);
        // compute output row
        row = reorder_dims(row, {perm.begin(), perm.end()});

        return op;
    }

    std::pair<instruction_ref, instruction_ref>
    apply_broadcast_op(const onnx_parser::node_info& info,
                       instruction_ref opl,
                       instruction_ref opr,
                       const std::vector<int>& common_labels) const
    {
        std::pair<instruction_ref, instruction_ref> ret;

        auto llens = opl->get_shape().lens();
        auto rlens = opr->get_shape().lens();

        bool lbc = false;
        bool rbc = false;
        for(auto l : common_labels)
        {
            if(llens[l] == 1 and rlens[l] == 1)
                continue;

            if(llens[l] == 1)
            {
                lbc      = true;
                llens[l] = rlens[l];
            }

            if(rlens[l] == 1)
            {
                rbc      = true;
                rlens[l] = llens[l];
            }
        }

        if(lbc)
            opl = info.add_instruction(make_op("multibroadcast", {{"out_lens", llens}}), opl);
        if(rbc)
            opr = info.add_instruction(make_op("multibroadcast", {{"out_lens", rlens}}), opr);

        ret.first  = opl;
        ret.second = opr;
        return ret;
    }

    instruction_ref apply_reduce_sum_op(const onnx_parser::node_info& info,
                                        instruction_ref op,
                                        const std::vector<int>& axes,
                                        std::vector<int>& row) const
    {
        if(axes.empty())
            return op;

        for(int a : axes)
            row[a] = -1;

        return info.add_instruction(make_op("reduce_sum", {{"axes", axes}}), op);
    }

    // Utility

    int_mat make_matrix(int cur_pair, int cols, int fill_value) const
    {
        return {static_cast<size_t>(cur_pair), std::vector<int>(cols, fill_value)};
    }

    std::vector<int> extract_column(int_mat map_mat, int col_idx, int row_begin, int row_end) const
    {
        std::vector<int> ret;
        ret.reserve(row_end - row_begin);

        std::transform(map_mat.begin() + row_begin,
                       map_mat.begin() + row_end,
                       std::back_inserter(ret),
                       [col_idx](const auto& x) { return x[col_idx]; });

        return ret;
    }

    std::vector<int> set_union(const std::vector<int>& lhs, const std::vector<int>& rhs) const
    {
        std::vector<int> ret;
        std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    std::vector<int> set_difference(const std::vector<int>& lhs, const std::vector<int>& rhs) const
    {
        std::vector<int> ret;
        std::set_difference(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    // Equivalent to numpy.arange without the step parameter
    std::vector<int> arange(int start_value, int end_value) const
    {
        std::vector<int> ret(end_value - start_value);
        std::iota(ret.begin(), ret.end(), start_value);
        return ret;
    }

    template <class... Vecs>
    std::decay_t<std::tuple_element_t<0, std::tuple<Vecs...>>> concat_vectors(Vecs&&... vecs) const
    {
        size_t reserve_size = 0u;
        ([&](auto&& vec) { reserve_size += vec.size(); }(std::forward<Vecs>(vecs)), ...);

        std::decay_t<std::tuple_element_t<0, std::tuple<Vecs...>>> ret;
        ret.reserve(reserve_size);

        ([&](auto&& vec) { ret.insert(ret.end(), vec.begin(), vec.end()); }(
             std::forward<Vecs>(vecs)),
         ...);

        return ret;
    }

    size_t calc_dim(const std::vector<int>& axes, const std::vector<size_t>& lens) const
    {
        return std::accumulate(
            axes.begin(), axes.end(), 1, [&](auto acc, auto axis) { return acc * lens[axis]; });
    };
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
