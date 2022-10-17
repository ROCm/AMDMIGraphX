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

        auto throw_shapes = [&]() {
            MIGRAPHX_THROW("PARSE_IF: " + info.name +
                           " then and else sub_graphs must have compatible shapes ");
        };

        if(then_out_shapes.size() != else_out_shapes.size())
        {
            throw_shapes();
        }

        // Add checks for each output shape
        for(int i = 0; i < then_out_shapes.size(); i++)
        {
            // Must have the same type for both if/else blocks by onnx spec
            if(then_out_shapes.at(i).type() != else_out_shapes.at(i).type())
            {
                MIGRAPHX_THROW("PARSE_IF: " + info.name +
                               " then and else sub_grahps must have same output type! " +
                               then_out_shapes.at(i).type_string() + " vs " +
                               else_out_shapes.at(i).type_string());
            }

            if(not then_out_shapes.at(i).dynamic() && not else_out_shapes.at(i).dynamic())
            {
                auto then_lens = then_out_shapes.at(i).lens();
                auto else_lens = else_out_shapes.at(i).lens();

                // Throw error if both branches have zero output shapes. Not possible for static
                // inputs
                if(then_lens.empty() && else_lens.empty())
                {
                    throw_shapes();
                }

                auto handle_empty_branch = [](module_ref& mdl, const shape& out_shape) {
                    auto outline_ins = mdl->add_outline(out_shape);
                    mdl->replace_return({outline_ins});
                    return out_shape.lens();
                };

                // Handle one empty branch by setting output identical to the other
                // need to update the then_shape before we do further checks
                if(then_lens.empty())
                {
                    then_lens = handle_empty_branch(then_mdl, else_out_shapes.at(i));
                }
                else if(else_lens.empty())
                {
                    else_lens = handle_empty_branch(else_mdl, then_out_shapes.at(i));
                }

                auto all_but_last_dims_equal = [](const std::vector<size_t>& lens_a,
                                                  const std::vector<size_t>& lens_b) {
                    if(lens_a.size() <= lens_b.size())
                    {
                        return std::equal(lens_a.begin(), lens_a.end(), lens_b.begin());
                    }
                    else
                    {
                        return std::equal(lens_b.begin(), lens_b.end(), lens_a.begin());
                    }
                };

                // check equivalent length dims, and (x1,x2,.., xn, 1) == (x1,x2,..,xn)
                int dim_delta = abs((static_cast<int>(then_lens.size() - else_lens.size())));

                if(dim_delta == 1)
                {
                    // make sure dims are equivalent in static shapes
                    if(not all_but_last_dims_equal(then_lens, else_lens))
                    {
                        throw_shapes();
                    }

                    auto last_then = then_lens.back();
                    auto last_else = else_lens.back();

                    auto unsqueeze_last_op = [](module_ref& mdl,
                                                const std::vector<size_t>& out_shape) {
                        auto convert_ins = mdl->add_instruction(
                            make_op("unsqueeze", {{"axes", {out_shape.size() - 1}}}),
                            std::prev(mdl->end())->inputs().front());
                        mdl->replace_return({convert_ins});
                        mdl->remove_instruction({std::prev(convert_ins)});
                    };

                    // Find which dim to unsqueeze
                    if((then_lens.size() < else_lens.size()) && (last_else == 1))
                    {
                        unsqueeze_last_op(then_mdl, else_lens);
                    }
                    else if((then_lens.size() > else_lens.size()) && (last_then == 1))
                    {
                        unsqueeze_last_op(else_mdl, then_lens);
                    }
                }
                else if(dim_delta > 1)
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
