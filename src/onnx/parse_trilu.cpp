/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_trilu : op_parser<parse_trilu>
{
    std::vector<op_desc> operators() const { return {{"Trilu"}}; }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto input_shape = args[0]->get_shape();
        auto input_lens  = input_shape.lens();

        size_t num_rows = *(input_lens.rbegin() + 1);
        size_t num_cols = input_lens.back();
        size_t k = 0;

        if(args.size() > 1)
        {
            auto arg_k = args[1]->eval();
            check_arg_empty(arg_k, "PARSE_TRILU: dynamic k not supported");
            k = arg_k.at<size_t>();
        }

        shape::type_t output_type = args[0]->get_shape().type();
        

        std::vector<char> mask_mat(num_rows * num_cols, 1);
        for(size_t i = 0; i < num_rows; i++)
        {
            for(size_t j= 0; j < k; j++)
            {
                mask_mat[i*num_cols + j] = 0;
            }
            k++;
        }
        auto mask = info.add_literal(
            migraphx::literal{migraphx::shape{output_type, input_lens}, mask_mat});

        return info.add_instruction(make_op("mul"), mask, args[0]);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
