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
#include <migraphx/onnx/reshape_helpers.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/reshape_dims.hpp>

#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_reshape : op_parser<parse_reshape>
{
    std::vector<op_desc> operators() const { return {{"Reshape"}}; }

    static instruction_ref add_evaluated_reshape(const onnx_parser::node_info& info,
                                                 instruction_ref input,
                                                 const std::vector<int64_t>& dims)
    {
        if(not input->get_shape().symbolic())
            return info.add_instruction(make_op("reshape", {{"dims", dims}}), input);
        auto shape_sources = info.mod->get_parameters();
        if(std::find(shape_sources.begin(), shape_sources.end(), input) == shape_sources.end())
            shape_sources.push_back(input);
        auto result = try_add_runtime_reshape(
            info, input, std::vector<dim_like>{dims.begin(), dims.end()}, shape_sources);
        if(not result.has_value())
            MIGRAPHX_THROW("Reshape: symbolic dimensions require direct runtime shape sources");
        return *result;
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> dims;

        if(args.size() == 1)
        {
            literal s = parser.parse_value(info.attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
            return add_evaluated_reshape(info, args[0], dims);
        }
        else
        {
            // 2 inputs
            auto s = args[1]->eval();
            if(s.empty())
            {
                // arg[1] not eval-able
                const auto symbolic_dims = args[1]->sym_eval();
                const auto& input_shape  = args[0]->get_shape();
                if(not symbolic_dims.empty() and
                   (not input_shape.dynamic() or input_shape.symbolic()))
                {
                    auto shape_sources = info.mod->get_parameters();
                    if(std::find(shape_sources.begin(), shape_sources.end(), args[0]) ==
                       shape_sources.end())
                        shape_sources.push_back(args[0]);
                    auto result = try_add_runtime_reshape(
                        info, args[0], symbolic_dims.get().to_vector(), shape_sources);
                    if(result.has_value())
                        return *result;
                }
                auto alloc_ins = info.add_instruction(
                    make_op("allocate", {{"buf_type", input_shape.type()}}), args[1]);
                return info.add_instruction(make_op("reshape"), args[0], alloc_ins);
            }
            else
            {
                s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
                return add_evaluated_reshape(info, args[0], dims);
            }
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
