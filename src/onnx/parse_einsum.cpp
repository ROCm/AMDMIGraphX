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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_einsum : op_parser<parse_einsum>
{
    using string_vec   = std::vector<std::string>;
    using int_vec      = std::vector<int>;
    using int_mat      = std::vector<std::vector<int>>;
    using char_int_map = std::map<char, int>;

    std::vector<op_desc> operators() const { return {{"Einsum"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(not contains(info.attributes, "equation"))
            MIGRAPHX_THROW("Equation attribute is required");
        std::string equation = info.attributes.at("equation").s();

        auto [terms, unique_labels, ellipses_ndim] = analyze_equation(equation, args);
        auto mat        = make_mapping_matrix(terms, unique_labels, ellipses_ndim);
        auto duplicates = look_for_duplicates(terms);
        int_mat rows    = full(2, mat[0].size(), -1);

        instruction_ref op;
        std::optional<instruction_ref> last_op;
        for(auto arg_idx = 0; arg_idx < args.size(); ++arg_idx)
        {
            op      = args[arg_idx];
            rows[1] = mat[arg_idx];

            auto duplicate = duplicates[arg_idx];
            op             = preprocess_input(info, op, duplicate, mat, arg_idx, rows);

            if(last_op)
                op = process_pair(info, *last_op, op, mat, arg_idx, rows);

            last_op = op;
            rows[0] = rows[1];
        }

        return finalize_output(info, op, mat, rows);
    }

    instruction_ref process_pair(const onnx_parser::node_info& info,
                                 instruction_ref op1,
                                 instruction_ref op2,
                                 const int_mat& mat,
                                 size_t input_idx,
                                 int_mat& rows) const
    {
        // Label is present in current two terms, but not in the remainder of the equation
        int_vec common_axes;
        // Label is present in only left term or both terms and somewhere in the remainder
        // of the equation
        int_vec left_only;
        // Label is present in only right term or both terms and somewhere in the remainder
        // of the equation
        int_vec right_only;
        int_vec sum_axes;
        int_vec unsqueezed_axes;

        auto not_neg_one = [](auto i) { return i != -1; };
        for(int d = 0; d < mat[0].size(); ++d)
        {
            // There is no -1 in the column, for the current two rows
            // The label is present in both rows
            if(all_of(extract_column(rows, d, 0, rows.size()), not_neg_one))
            {
                // There is at least 1 element that is not -1, for the remaining rows of the
                // matrix.
                // The label is present in at least one of the subsequent rows
                if(any_of(extract_column(mat, d, input_idx + 1, mat.size()), not_neg_one))
                    common_axes.push_back(d);
                else
                    sum_axes.push_back(d);
            }
            // The label is missing in one or both of the rows
            else
            {
                if(rows[0][d] >= 0)
                    left_only.push_back(d);
                else if(rows[1][d] >= 0)
                    right_only.push_back(d);
                else
                    unsqueezed_axes.push_back(d);
            }
        }

        auto batch_axes = set_union(common_axes, unsqueezed_axes);
        auto perm       = concat_vectors(batch_axes, left_only, right_only, sum_axes);
        op1             = apply_transpose_op(info, op1, perm, rows[0]);
        op2             = apply_transpose_op(info, op2, perm, rows[1]);

        auto new_batch_axes = arange(0, batch_axes.size());
        auto new_sum_axes   = arange(rows[0].size() - sum_axes.size(), rows[0].size());
        auto perm_for_side  = [&](const auto& labels) {
            int_vec ret;
            for(int i = 0; i < perm.size(); ++i)
                if(contains(labels, perm[i]))
                    ret.push_back(i);
            return ret;
        };
        auto perm_left  = perm_for_side(set_union(left_only, common_axes));
        auto perm_right = perm_for_side(set_union(right_only, common_axes));
        auto op =
            batch_dot(info, rows, op1, op2, new_batch_axes, new_sum_axes, perm_left, perm_right);

        auto ordered_axes = concat_vectors(batch_axes, left_only, right_only, sum_axes);
        perm              = make_ordered_permutation(ordered_axes);

        return apply_transpose_op(info, op, perm, rows[1]);
    }

    instruction_ref preprocess_input(const onnx_parser::node_info& info,
                                     instruction_ref op,
                                     const std::map<char, int_vec>& duplicates,
                                     const int_mat& mat,
                                     size_t input_idx,
                                     int_mat& rows) const
    {
        if(not duplicates.empty())
        {
            std::vector<std::tuple<int, int_vec>> diag;
            for(auto [_, v] : duplicates)
                if(v.size() > 1)
                    diag.push_back({v[0], v});

            op = apply_diagonal(info, rows, op, diag);
        }

        // Transpose so the labels in the term are ordered alphabetically
        op = unsqueeze_transpose(info, rows, op);

        int_vec red;
        for(int d = 0; d < mat[0].size(); ++d)
        {
            bool all_neg_one = all_of(extract_column(mat, d, input_idx + 1, mat.size()),
                                      [](auto i) { return i == -1; });
            if(all_neg_one and rows[1][d] != -1 and rows[0][d] == -1)
                red.push_back(d);
        }

        return apply_reduce_sum_op(info, op, red, rows[1]);
    }

    instruction_ref finalize_output(const onnx_parser::node_info& info,
                                    instruction_ref op,
                                    const int_mat& mat,
                                    int_mat& rows) const
    {
        if(any_of(mat.back(), [](auto i) { return i >= 0; }))
        {
            rows[1] = mat.back();
            int_vec red;
            for(int d = 0; d < mat[0].size(); ++d)
            {
                if(rows[0][d] > 0 and rows[1][d] == -1)
                    red.push_back(d);
                else if(rows[0][d] == -1 and rows[1][d] >= 0)
                    MIGRAPHX_THROW("Issue in equation");
            }

            op = apply_reduce_sum_op(info, op, red, rows[1]);
        }

        return transpose_squeeze(info, rows, op, mat.back());
    }

    instruction_ref apply_diagonal(const onnx_parser::node_info& info,
                                   int_mat& rows,
                                   instruction_ref op,
                                   std::vector<std::tuple<int, int_vec>> diag) const
    {
        if(diag.size() != 1)
            MIGRAPHX_THROW("Not implemented with more than one duplicated label");

        auto axis = std::get<0>(diag[0]);
        auto axes = std::get<1>(diag[0]);

        int_vec batch_axes;
        for(int i = 0; i < rows[0].size(); ++i)
            if(not contains(axes, i))
                batch_axes.push_back(i);

        auto min_axes = *(std::min_element(axes.begin(), axes.end()));
        if(not all_of(batch_axes, [=](int ba) { return ba < min_axes; }))
            MIGRAPHX_THROW("Currently batch axes have to be partitioned to the left");

        auto op_shape = op->get_shape().lens();

        if(not all_of(axes, [op_shape, axis](int a) { return op_shape[axis] == op_shape[a]; }))
            MIGRAPHX_THROW("All duplicated indices have to be the same dimension");

        size_t batch_size =
            std::accumulate(batch_axes.begin(), batch_axes.end(), 1, [&](auto acc, auto dim) {
                return acc *= op_shape[dim];
            });

        std::vector<size_t> indices;
        for(int batch = 0; batch < batch_size; ++batch)
        {
            for(int i = 0; i < op_shape[axis]; ++i)
            {
                std::vector<size_t> index(axes.size(), static_cast<size_t>(i));
                indices.insert(indices.end(), index.begin(), index.end());
            }
        }

        std::vector<size_t> lens{op_shape[axis], axes.size()};
        if(batch_size > 1)
            lens.insert(lens.begin(), batch_size);

        auto indices_arg = info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, lens}, indices});

        op = info.add_instruction(
            migraphx::make_op("gathernd", {{"batch_dims", batch_axes.size()}}), op, indices_arg);
        // compute output row
        int_vec to_remove;
        for(auto [choice, choices] : diag)
        {
            std::copy_if(choices.begin(),
                         choices.end(),
                         std::back_inserter(to_remove),
                         [ch = choice](const auto& el) { return el != ch; });

            for(auto& r : rows[1])
                if(contains(choices, r))
                    r = choice;
        }

        std::sort(to_remove.begin(), to_remove.end());
        for(auto t : to_remove)
        {
            for(auto& r : rows[1])
            {
                if(r == t)
                    MIGRAPHX_THROW("Unexpected result");

                if(r > t)
                    r -= 1;
            }
        }

        return op;
    }

    instruction_ref
    unsqueeze_transpose(const onnx_parser::node_info& info, int_mat& rows, instruction_ref op) const
    {
        int_vec unsq_axes;
        std::vector<std::tuple<int, int>> perm;

        for(auto i = 0; i < rows[1].size(); ++i)
        {
            if(rows[1][i] == -1)
                unsq_axes.push_back(i);
            else
                perm.push_back({rows[1][i], i});
        }
        op = info.add_instruction(make_op("unsqueeze", {{"axes", unsq_axes}}), op);

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        int_vec new_perm(rows[1].size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        for(auto i = 0, p = 0; i < rows[1].size(); ++i)
        {
            if(rows[1][i] == -1)
                continue;

            new_perm[std::get<1>(perm[p++])] = i;
        }

        op = apply_transpose_op(info, op, new_perm, rows[1]);

        return op;
    }

    instruction_ref transpose_squeeze(const onnx_parser::node_info& info,
                                      int_mat& rows,
                                      instruction_ref op,
                                      int_vec row_output) const
    {
        std::vector<std::tuple<int, int>> perm;
        int_vec sq;

        for(auto i = 0; i < row_output.size(); ++i)
        {
            if(row_output[i] == -1)
                sq.push_back(i);
            else
                perm.push_back({row_output[i], i});
        }

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        int_vec new_perm(rows[1].size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        for(auto i = 0, p = 0; i < row_output.size(); ++i)
        {
            if(row_output[i] == -1)
                continue;

            new_perm[i] = std::get<1>(perm[p++]);
        }

        op = apply_transpose_op(info, op, new_perm, rows[1]);

        if(not sq.empty())
        {
            op = info.add_instruction(make_op("squeeze", {{"axes", sq}}), op);
            // compute output row
            for(int a : sq)
                rows[1][a] = -1;
        }

        return op;
    }

    instruction_ref batch_dot(const onnx_parser::node_info& info,
                              int_mat& rows,
                              instruction_ref op1,
                              instruction_ref op2,
                              const int_vec& batch_axes,
                              const int_vec& sum_axes,
                              const int_vec& perm_left,
                              const int_vec& perm_right) const
    {
        auto common_labels = set_union(batch_axes, sum_axes);
        std::tie(op1, op2) = apply_broadcast_op(info, op1, op2, common_labels);

        auto op1_shape = op1->get_shape().lens();
        auto op2_shape = op2->get_shape().lens();

        auto calc_dim = [](const auto& axes, const auto& lens) {
            return std::accumulate(
                axes.begin(), axes.end(), 1, [&](auto acc, auto l) { return acc *= lens[l]; });
        };
        std::array<int, 3> dims1{
            calc_dim(batch_axes, op1_shape), -1, calc_dim(sum_axes, op1_shape)};
        std::array<int, 3> dims2{
            calc_dim(batch_axes, op2_shape), -1, calc_dim(sum_axes, op2_shape)};

        op1 = info.add_instruction(make_op("reshape", {{"dims", dims1}}), op1);
        op2 = info.add_instruction(make_op("reshape", {{"dims", dims2}}), op2);
        op2 = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), op2);
        instruction_ref dot = info.add_instruction(make_op("dot"), op1, op2);

        int_vec new_lens;

        std::transform(batch_axes.begin(),
                       batch_axes.end(),
                       std::back_inserter(new_lens),
                       [&](auto a) { return std::max(op1_shape[a], op2_shape[a]); });

        for(int i : perm_left)
            if(not contains(batch_axes, i))
                new_lens.push_back(op1_shape[i]);

        for(int i : perm_right)
            if(not contains(batch_axes, i))
                new_lens.push_back(op2_shape[i]);

        while(new_lens.size() < op1_shape.size())
            new_lens.push_back(1);

        auto op = info.add_instruction(make_op("reshape", {{"dims", new_lens}}), dot);
        // compute output row
        std::transform(
            rows[0].begin(), rows[0].end(), rows[1].begin(), rows[1].begin(), [](int l, int r) {
                return std::max(l, r);
            });

        for(int a : sum_axes)
            if(not contains(perm_right, a))
                rows[1][a] = -1;

        return op;
    }

    bool is_transpose_identity(int_vec perm) const
    {
        for(auto i = 0u; i < perm.size(); ++i)
            if(perm[i] != i)
                return false;

        return true;
    }

    int_mat full(int rows, int cols, int fill_value) const
    {
        int_mat ret(rows);
        for(auto& row : ret)
            for(int i = 0; i < cols; ++i)
                row.push_back(fill_value);

        return ret;
    }

    int_vec extract_column(int_mat mat, int col_idx, int row_begin, int row_end) const
    {
        int_vec ret;
        ret.reserve(row_end - row_begin);

        for(int i = row_begin; i < row_end; ++i)
            ret.push_back(mat[i][col_idx]);

        return ret;
    }

    int_vec set_union(const int_vec& lhs, const int_vec& rhs) const
    {
        int_vec ret;
        std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    int_vec set_intersection(const int_vec& lhs, const int_vec& rhs) const
    {
        int_vec ret;
        std::set_intersection(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    template <typename... Vecs>
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

    std::tuple<string_vec, std::string, size_t>
    analyze_equation(std::string_view equation, const std::vector<instruction_ref>& args) const
    {
        std::tuple<string_vec, std::string, size_t> ret;
        auto& [terms, unique_labels, ellipses_ndim] = ret;

        auto [input_terms, output_term, label_count, explicit_form] = parse_equation(equation);

        ellipses_ndim = validate_input_terms(input_terms, args);
        if(not output_term.empty())
            validate_output_term(output_term, label_count, ellipses_ndim);
        else if(not explicit_form)
            output_term = generate_output_term(label_count, ellipses_ndim);

        terms = std::move(input_terms);
        terms.emplace_back(std::move(output_term));
        for(auto [l, _] : label_count)
            unique_labels += l;

        return ret;
    }

    int_mat make_mapping_matrix(const string_vec& terms,
                                std::string_view unique_labels,
                                size_t ellipses_ndim) const
    {
        std::map<char, int> label_to_column;
        for(auto i = 0; i < unique_labels.size(); ++i)
            label_to_column[unique_labels[i]] = i;

        int_mat mat = full(terms.size(), unique_labels.size() + ellipses_ndim, -1);

        for(auto i = 0; i < terms.size(); ++i)
        {
            const auto& term = terms[i];
            int col_id       = 0;
            for(auto l : term)
            {
                if(l == '*')
                {
                    std::iota(mat[i].end() - ellipses_ndim, mat[i].end(), col_id);
                    col_id += ellipses_ndim;
                }
                else
                    mat[i][label_to_column[l]] = col_id++;
            }
        }

        return mat;
    }

    std::vector<std::map<char, int_vec>> look_for_duplicates(const string_vec& terms) const
    {
        std::vector<std::map<char, int_vec>> duplicates;
        for(auto term : terms)
        {
            if(term.size() == std::set<char>(term.begin(), term.end()).size())
            {
                duplicates.push_back({});
                continue;
            }

            std::map<char, int_vec> counts;
            int i = 0;
            for(char c : term)
            {
                counts[c].push_back(i++);
            }
            duplicates.push_back(counts);
        }

        return duplicates;
    }

    std::tuple<std::vector<std::string>, std::string, std::map<char, int>, bool>
    parse_equation(std::string_view equation) const
    {
        std::tuple<std::vector<std::string>, std::string, std::map<char, int>, bool> ret;
        // cppcheck-suppress variableScope
        auto& [input_terms, output_term, label_count, explicit_form] = ret;

        std::string term;
        bool has_ellipsis = false;
        explicit_form     = false;

        for(int i = 0; i < equation.size(); ++i)
        {
            const char c = equation[i];
            switch(c)
            {
            case ' ': break;
            case '-':
                if(explicit_form)
                    MIGRAPHX_THROW("Einsum equation has multiple '->' symbols");
                if(i + 1 >= equation.size() or equation[i + 1] != '>')
                    MIGRAPHX_THROW("Invalid '->' in einsum equation");

                ++i;
                explicit_form = true;
                [[fallthrough]];
            case ',':
                has_ellipsis = false;
                input_terms.emplace_back(term);
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
                if(not explicit_form)
                    ++label_count[c];
            }
        }

        if(explicit_form)
            output_term = term;
        else
            input_terms.push_back(term);

        return ret;
    }

    std::string generate_output_term(const char_int_map& label_count, size_t ellipsis_ndim) const
    {
        std::string output_term = ellipsis_ndim == 0 ? "" : "*";
        for(const auto [label, count] : label_count)
            if(count == 1)
                output_term += label;

        return output_term;
    }

    void validate_output_term(std::string_view output_term,
                              const char_int_map& label_count,
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

    size_t validate_input_terms(const string_vec& input_terms,
                                const std::vector<instruction_ref>& args) const
    {
        if(input_terms.size() != args.size())
            MIGRAPHX_THROW(
                "Number of terms in the input equation - " + std::to_string(input_terms.size()) +
                " does not match the number of input tensors " + std::to_string(args.size()));

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
                    auto ellipses_dims = rank - term.size() + 1;
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

    std::pair<instruction_ref, instruction_ref>
    apply_broadcast_op(const onnx_parser::node_info& info,
                       instruction_ref opl,
                       instruction_ref opr,
                       const int_vec& common_labels) const
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

    instruction_ref apply_transpose_op(const onnx_parser::node_info& info,
                                       instruction_ref op,
                                       const int_vec& perm,
                                       int_vec& row) const
    {
        if(is_transpose_identity(perm))
            return op;

        op = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op);
        // compute output row
        auto cpy = row;
        for(auto i = 0u; i < perm.size(); ++i)
            row[i] = cpy[perm[i]];

        return op;
    }

    instruction_ref apply_reduce_sum_op(const onnx_parser::node_info& info,
                                        instruction_ref op,
                                        const int_vec& axes,
                                        int_vec& row) const
    {
        if(axes.empty())
            return op;

        for(int a : axes)
            row[a] = -1;

        return info.add_instruction(make_op("reduce_sum", {{"axes", axes}}), op);
    }

    int_vec make_ordered_permutation(const int_vec& axes) const
    {
        int_vec ret(axes.size());
        for(auto i = 0; i < axes.size(); ++i)
            ret[axes[i]] = i;

        return ret;
    }

    int_vec arange(int start_value, int end_value) const
    {
        int_vec ret(end_value - start_value);
        std::iota(ret.begin(), ret.end(), start_value);
        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
