/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_trilu : op_parser<parse_trilu>
{
    std::vector<op_desc> operators() const { return {{"Trilu"}}; }

    static std::vector<std::int64_t> make_diagonal_values(std::size_t num_rows,
                                                          std::size_t num_cols)
    {
        std::vector<std::int64_t> result(num_rows * num_cols);
        auto row_begin               = result.begin();
        std::int64_t first_row_value = 0;
        for(std::size_t row = 0; row < num_rows; ++row)
        {
            const auto row_end = row_begin + num_cols;
            std::iota(row_begin, row_end, first_row_value);
            row_begin = row_end;
            --first_row_value;
        }
        return result;
    }

    static std::vector<bool>
    make_mask_values(const std::vector<std::int64_t>& diagonal, std::int64_t k, bool upper)
    {
        std::vector<bool> result(diagonal.size());
        std::transform(diagonal.begin(), diagonal.end(), result.begin(), [k, upper](auto value) {
            return upper ? value >= k : value <= k;
        });
        return result;
    }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        const auto input_shape = args[0]->get_shape();
        assert(input_shape.ndim() >= 2);
        std::int64_t k = 0;
        bool upper     = true;

        if(args.size() > 1)
        {
            const auto arg_k = args[1]->eval();
            check_arg_empty(arg_k, "PARSE_TRILU: dynamic k not supported");
            k = arg_k.at<std::int64_t>();
        }

        if(contains(info.attributes, "upper"))
        {
            upper = info.attributes.at("upper").i() != 0;
        }

        const auto output_type = input_shape.type();

        std::size_t num_rows = 0;
        std::size_t num_cols = 0;
        bool fixed_matrix    = not input_shape.dynamic();
        if(fixed_matrix)
        {
            const auto& input_lens = input_shape.lens();
            num_rows               = *(input_lens.rbegin() + 1);
            num_cols               = input_lens.back();
        }
        else
        {
            const auto& input_dims = input_shape.dyn_dims();
            const auto& rows       = *(input_dims.rbegin() + 1);
            const auto& cols       = input_dims.back();
            fixed_matrix           = rows.is_fixed() and cols.is_fixed();
            if(fixed_matrix)
            {
                num_rows = rows.get_interval().min;
                num_cols = cols.get_interval().min;
            }
        }

        if(fixed_matrix)
        {
            const auto diagonal_values = make_diagonal_values(num_rows, num_cols);
            const auto mask_values     = make_mask_values(diagonal_values, k, upper);
            const auto mask =
                info.add_literal(literal{shape{output_type, {num_rows, num_cols}}, mask_values});
            return info.add_broadcastable_binary_op("mul", mask, args[0]);
        }

        if(not input_shape.symbolic())
            MIGRAPHX_THROW("PARSE_TRILU: range-dynamic matrix dimensions not supported");

        const auto rank        = input_shape.ndim();
        const auto matrix_dims = info.add_instruction(
            make_op("dimensions_of", {{"start", rank - 2}, {"end", rank}}), args[0]);
        const auto rows = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), matrix_dims);
        const auto cols = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), matrix_dims);

        const auto zero =
            info.add_literal(literal{shape{shape::int64_type, {1}, {0}}, {std::int64_t{0}}});
        const auto one =
            info.add_literal(literal{shape{shape::int64_type, {1}, {0}}, {std::int64_t{1}}});
        const auto max_output  = std::numeric_limits<int>::max();
        const auto& input_dims = input_shape.dyn_dims();
        const auto matrix_shape =
            std::vector<shape::dynamic_dimension>{*(input_dims.rbegin() + 1), input_dims.back()};
        const auto row_range =
            make_op("dynamic_range",
                    {{"max_output", max_output}, {"output_dim", to_value(matrix_shape.front())}});
        const auto col_range =
            make_op("dynamic_range",
                    {{"max_output", max_output}, {"output_dim", to_value(matrix_shape.back())}});
        auto row_indices = info.add_instruction(row_range, zero, rows, one);
        auto col_indices = info.add_instruction(col_range, zero, cols, one);
        row_indices      = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), row_indices);
        col_indices      = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), col_indices);

        auto col_matrix = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(matrix_shape)}}),
            col_indices,
            row_indices);
        auto row_matrix = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(matrix_shape)}}),
            row_indices,
            col_matrix);
        auto diagonal = info.add_instruction(make_op("sub"), col_matrix, row_matrix);
        auto k_input  = info.add_literal(literal{shape{shape::int64_type, {1}, {0}}, {k}});
        k_input       = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(matrix_shape)}}),
            k_input,
            diagonal);
        auto excluded =
            info.add_instruction(make_op(upper ? "less" : "greater"), diagonal, k_input);
        if(excluded->get_shape().type() != shape::bool_type)
        {
            excluded = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                            excluded);
        }
        auto mask = info.add_instruction(make_op("not"), excluded);
        if(output_type != shape::bool_type)
        {
            mask = info.add_instruction(make_op("convert", {{"target_type", output_type}}), mask);
        }
        mask = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(input_shape.dyn_dims())}}),
            mask,
            args[0]);
        return info.add_instruction(make_op("mul"), args[0], mask);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
