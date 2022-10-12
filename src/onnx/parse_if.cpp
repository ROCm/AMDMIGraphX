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
#include <migraphx/instruction_ref.hpp>
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/reduce_dims.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_if : op_parser<parse_if>
{
    std::vector<op_desc> operators() const { return {{"If"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        const auto& then_graph = info.attributes.at("then_branch").g();
        const auto& else_graph = info.attributes.at("else_branch").g();

        if(args.front()->get_shape().elements() != 1)
        {
            MIGRAPHX_THROW("PARSE_IF: " + info.name +
                           " condition input can have only one element!");
        }

        std::string then_name = info.name + "_if";
        module_ref then_mdl   = parser.prog.create_module(then_name);

        std::string else_name = info.name + "_else";
        module_ref else_mdl   = parser.prog.create_module(else_name);

        // parse the then sub_graph
        parser.parse_graph(then_mdl, then_graph);

        // parse_the else sub_graph
        parser.parse_graph(else_mdl, else_graph);

        auto then_out_shapes = then_mdl->get_output_shapes();
        auto else_out_shapes = else_mdl->get_output_shapes();

        assert(then_out_shapes.size() == else_out_shapes.size());

        // Must have the same type for both if/else blocks by onnx spec
        // Add exception for empty constant scalars
        if(then_out_shapes.at(0).type() != else_out_shapes.at(0).type())
        {
            MIGRAPHX_THROW("PARSE_IF: " + info.name +
                           " then and else sub_grahps must have same output type! " +
                           then_out_shapes.at(0).type_string() + " vs " +
                           else_out_shapes.at(0).type_string());
        }

        if(not then_out_shapes.at(0).dynamic() && not else_out_shapes.at(0).dynamic())
        {
            auto then_shape   = then_out_shapes.at(0).lens();
            auto else_shape   = else_out_shapes.at(0).lens();
            auto throw_shapes = [&]() {
                MIGRAPHX_THROW("PARSE_IF: " + info.name +
                               " then and else sub_graphs must compatible shapes ");
            };

            // Throw error if both branches have zero output shapes. Not possible for static inputs
            if(then_shape.size() == 0 && else_shape.size() == 0)
            {
                throw_shapes();
            }

            // Handle one empty branch by setting output identical to the other
            // need to update the then_shape before we do further checks
            if(then_shape.size() == 0)
            {
                auto convert_ins = then_mdl->add_outline(else_out_shapes.at(0));
                then_mdl->replace_return({convert_ins});
                then_shape = else_shape;
            }
            else if(else_shape.size() == 0)
            {
                auto convert_ins = else_mdl->add_outline(then_out_shapes.at(0));
                else_mdl->replace_return({convert_ins});
                else_shape = then_shape;
            }
            else
            {
                // check equivilant length dims, and (x1,x2,.., xn, 1) == (x1,x2,..,xn)
                int dim_delta = abs((static_cast<int>(then_shape.size() - else_shape.size())));

                if(dim_delta <= 1)
                {
                    // make sure dims are equivalent in static shapes
                    if(not equal(then_shape.begin(), then_shape.end(), else_shape.begin()) &&
                       not equal(else_shape.begin(), else_shape.end(), then_shape.begin()))
                    {
                        throw_shapes();
                    }

                    // Find which dim to pad
                    if(then_shape.size() < else_shape.size())
                    {
                        auto last_else = *(--(else_shape.end()));

                        if(last_else <= 1)
                        {
                            auto convert_ins = then_mdl->add_outline(else_out_shapes.at(0));
                            then_mdl->replace_return({convert_ins});
                        }
                    }
                    else
                    {
                        auto last_then = *(--(then_shape.end()));

                        if(last_then <= 1)
                        {
                            auto convert_ins = else_mdl->add_outline(then_out_shapes.at(0));
                            else_mdl->replace_return({convert_ins});
                        }
                    }
                }
                else
                {
                    throw_shapes();
                }
            }
        }

        auto if_ret = info.add_instruction(make_op("if"), args, {then_mdl, else_mdl});
        auto out_s  = if_ret->get_shape();
        assert(out_s.type() == shape::tuple_type);

        const auto& vec_shapes = out_s.sub_shapes();
        std::vector<instruction_ref> out_inss;
        for(std::size_t i = 0; i < vec_shapes.size(); ++i)
        {
            auto ret = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), if_ret);
            out_inss.push_back(ret);
        }

        return out_inss;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
