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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {
struct parse_array_feature_extractor : op_parser<parse_array_feature_extractor>
{
    std::vector<op_desc> operators() const { return {{"ArrayFeatureExtractor"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                            const onnx_parser& /*parser*/,
                            onnx_parser::node_info info,
                            std::vector<instruction_ref> args) const
    {
        auto x = info.make_contiguous(args[0]);
        auto y  = info.make_contiguous(args[1]);
        auto data_s = x->get_shape();
        auto ind_s  = y->get_shape();

        auto ndim = data_s.ndim();
        if(ndim == 0){
            MIGRAPHX_THROW("PARSE_ARRAY_FEATURE_EXTRACTOR: input data must have at least 1 dimension");
        }
        auto axis = static_cast<int64_t>(ndim - 1);
        auto op = make_op("gather", {{"axis", axis}});
        return info.add_instruction(op, x, y);
    }
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx