/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <algorithm>
#include <map>
#include <optional>
#include <migraphx/sym.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_if : op_parser<parse_if>
{
    std::vector<op_desc> operators() const { return {{"If"}}; }

    // Only the returned values are made standard; other users of a returned value, such as a
    // Shape of it, keep reading it
    static void make_standard_return(module_ref mm)
    {
        auto returns = mm->get_returns();
        std::transform(returns.begin(), returns.end(), returns.begin(), [&](auto r) {
            if(r->get_shape().standard())
                return r;
            return mm->insert_instruction(std::prev(mm->end()), make_op("contiguous"), r);
        });
        mm->replace_return(returns);
    }

    // The output shapes when the branches return symbolic shapes that differ on some axes: such
    // an axis gets a new symbol covering both. Nothing when the shapes are equal or not symbolic.
    static std::optional<std::vector<shape>>
    differing_output_shapes(onnx_parser& parser,
                            const onnx_parser::node_info& info,
                            const std::vector<shape>& then_shapes,
                            const std::vector<shape>& else_shapes)
    {
        if(then_shapes == else_shapes or then_shapes.size() != else_shapes.size())
            return std::nullopt;
        auto symbolic = [](const shape& s) { return s.symbolic(); };
        if(not std::any_of(then_shapes.begin(), then_shapes.end(), symbolic) and
           not std::any_of(else_shapes.begin(), else_shapes.end(), symbolic))
            return std::nullopt;
        std::vector<shape> result;
        // Outputs of the same pair of lengths, such as every layer of a KV cache, share a symbol
        std::map<std::pair<std::string, std::string>, shape::dynamic_dimension> symbols;
        for(std::size_t i = 0; i < then_shapes.size(); ++i)
        {
            const auto& t = then_shapes[i];
            const auto& e = else_shapes[i];
            if(t.type() != e.type() or t.ndim() != e.ndim() or (t.dynamic() and not t.symbolic()) or
               (e.dynamic() and not e.symbolic()))
                return std::nullopt;
            auto t_dims = t.to_symbolic().dyn_dims();
            auto e_dims = e.to_symbolic().dyn_dims();
            std::vector<shape::dynamic_dimension> dims;
            for(std::size_t axis = 0; axis < t_dims.size(); ++axis)
            {
                if(t_dims[axis] == e_dims[axis])
                {
                    dims.push_back(t_dims[axis]);
                    continue;
                }
                auto key   = std::make_pair(t_dims[axis].sym_expr.to_string(),
                                            e_dims[axis].sym_expr.to_string());
                auto found = symbols.find(key);
                if(found == symbols.end())
                {
                    auto ti = t_dims[axis].get_interval();
                    auto ei = e_dims[axis].get_interval();
                    found = symbols
                                .emplace(key,
                                         shape::dynamic_dimension{sym::var(
                                             parser.symbol_names.allocate(info.name),
                                             {std::min(ti.min, ei.min), std::max(ti.max, ei.max)})})
                                .first;
                }
                dims.push_back(found->second);
            }
            result.emplace_back(t.type(), dims);
        }
        return result;
    }

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

        // Fold instruction if condition is constant thus can be evaled
        // prior to inference
        if(args.front()->can_eval())
        {
            auto cond_arg = args.front()->eval();
            auto* mod     = info.mod;
            // then branch
            if(cond_arg.at<bool>())
            {
                return parser.parse_graph(mod, then_graph, true);
            }
            // else branch
            else
            {
                return parser.parse_graph(mod, else_graph, true);
            }
        }

        std::string then_name = info.name + "_if";
        module_ref then_mdl   = parser.prog.create_module(then_name);

        std::string else_name = info.name + "_else";
        module_ref else_mdl   = parser.prog.create_module(else_name);

        // parse the then sub_graph
        (void)parser.parse_graph(then_mdl, then_graph);

        // parse_the else sub_graph
        (void)parser.parse_graph(else_mdl, else_graph);

        auto then_out_shapes = then_mdl->get_output_shapes();
        auto else_out_shapes = else_mdl->get_output_shapes();
        // With symbolic shapes the branches may return different lengths, such as a KV cache
        // with and without the past; the output then has a new symbol on such an axis
        auto dyn_shapes = differing_output_shapes(parser, info, then_out_shapes, else_out_shapes);
        if(dyn_shapes.has_value())
        {
            make_standard_return(then_mdl);
            make_standard_return(else_mdl);
            auto if_ret = info.add_instruction(
                make_op("if", {{"output_dyn_shapes", to_value(shape{*dyn_shapes})}}),
                args,
                {then_mdl, else_mdl});
            std::vector<instruction_ref> out_inss;
            for(std::size_t i = 0; i < dyn_shapes->size(); ++i)
                out_inss.push_back(
                    info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), if_ret));
            return out_inss;
        }
        if(not std::equal(then_out_shapes.begin(),
                          then_out_shapes.end(),
                          else_out_shapes.begin(),
                          else_out_shapes.end()))
        {
            if(not std::equal(then_out_shapes.begin(),
                              then_out_shapes.end(),
                              else_out_shapes.begin(),
                              else_out_shapes.end(),
                              [](const shape& then_s, const shape& else_s) {
                                  return then_s.as_standard() == else_s.as_standard();
                              }))
            {
                MIGRAPHX_THROW("PARSE_IF: " + info.name +
                               " then and else sub_graphs must have same output shapes! then: {" +
                               to_string_range(then_out_shapes) + "}, else: {" +
                               to_string_range(else_out_shapes) + "}");
            }
            make_standard_return(then_mdl);
            make_standard_return(else_mdl);
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
