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
#include "migraphx/argument.hpp"
#include "migraphx/errors.hpp"
#include "migraphx/instruction_ref.hpp"
#include "migraphx/iterator_for.hpp"
#include "migraphx/module_ref.hpp"
#include "migraphx/onnx/onnx_parser.hpp"
#include <algorithm>
#include <cstdint>
#include <ios>
#include <iterator>
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scan : op_parser<parse_scan>
{
    std::vector<op_desc> operators() const { return {{"Scan"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        check_for_required_attributes(info, {"body", "num_scan_inputs"});

        const auto& body_graph = info.attributes["body"].g();
        auto body              = parser.prog.create_module(info.name + "_scan");
        (void)parser.parse_graph(body, body_graph);

        auto body_outs = body->get_returns();
        const auto M   = info.attributes["num_scan_inputs"].i();
        const auto N   = args.size() - M;
        const auto K   = body_outs.size() - N;

        if(body->get_parameter_names().size() != N + M)
            MIGRAPHX_THROW("Lorem ipsum");

        const auto scan_input_axes = parse_axes(info, "scan_input_axes", M, args.begin() + N, 0);

        size_t num_iters = args[N]->get_shape().lens()[scan_input_axes[0]];
        for(auto i = 1; i < M; ++i)
        {
            if(args[i]->get_shape().lens()[scan_input_axes[i]] != num_iters)
                MIGRAPHX_THROW("Lorem ipsum");
        }

        const auto scan_input_directions = parse_dirs(info, "scan_input_directions", M);

        const auto scan_output_axes =
            parse_axes(info, "scan_output_axes", K, body_outs.begin() + N, 1);

        const auto scan_output_directions = parse_dirs(info, "scan_output_directions", K);

        // TODO check that alt_args shapes match body input parameter shapes

        modify_body(body, args, M, N, scan_input_axes, scan_input_directions);
        auto cond_lit = info.add_literal(literal{shape{shape::bool_type}, {true}});
        args.insert(args.begin(), cond_lit);
        auto max_iter_lit = info.add_literal(literal{shape{shape::int64_type}, {num_iters}});
        args.insert(args.begin(), max_iter_lit);

        auto loop =
            info.add_instruction(make_op("loop", {{"max_iterations", num_iters}}), args, {body});

        std::vector<instruction_ref> ret;
        ret.reserve(N + K);
        for(std::size_t i = 0; i < N; ++i)
        {
            auto r = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), loop);
            ret.push_back(r);
        }

        for(std::size_t i = N + M; i < N + M + K; ++i)
        {
            auto r         = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), loop);
            auto scan_axis = scan_output_axes[i - N - M];
            std::vector<int64_t> perm(r->get_shape().ndim(), 0);
            std::iota(perm.begin(), perm.end(), 0);
            std::copy(perm.begin() + 1, perm.begin() + 1 + scan_axis, perm.begin());
            perm[scan_axis] = 0;
            r = info.add_instruction(make_op("transpose", {{"permutation", perm}}), r);
            ret.push_back(r);
        }

        return ret;
    }

    void check_for_required_attributes(onnx_parser::node_info& info,
                                       std::vector<std::string> attribute_names) const
    {
        for(const auto& name : attribute_names)
            if(not contains(info.attributes, name))
                MIGRAPHX_THROW("Scan: " + name + " attribute required");
    }

    std::vector<int64_t> parse_vector_attribute(onnx_parser::node_info& info,
                                                const std::string& attr_name,
                                                size_t expected_size) const
    {
        if(not contains(info.attributes, attr_name))
            return {};

        std::vector<int64_t> res;
        auto&& attr = info.attributes[attr_name].ints();
        if(attr.size() != expected_size)
            MIGRAPHX_THROW("Scan: " + attr_name + " size is " + to_string(attr.size()) +
                           ", should be " + to_string(expected_size));
        res.assign(attr.begin(), attr.end());

        return res;
    }

    std::vector<int64_t>
    parse_dirs(onnx_parser::node_info& info, const std::string& name, size_t expected_size) const
    {
        auto dirs = parse_vector_attribute(info, name, expected_size);
        if(dirs.empty())
            return std::vector<int64_t>(expected_size, 0);

        if(any_of(dirs, [](auto i) { return i != 0 and i != 1; }))
            MIGRAPHX_THROW("Scan: " + name +
                           " may contain only 1s and 0s, actual values: " + to_string_range(dirs));

        return dirs;
    }

    int64_t normalize_axis(int64_t axis, int64_t rank) const
    {
        if(axis < -rank or axis >= rank)
            MIGRAPHX_THROW("Axis value {" + to_string(axis) + "} out of range [" +
                           to_string(-rank) + ", " + to_string(rank) + ")");

        return axis < 0 ? rank + axis : axis;
    }

    std::vector<int64_t> parse_axes(onnx_parser::node_info& info,
                                    const std::string& name,
                                    size_t expected_size,
                                    std::vector<instruction_ref>::iterator ins_begin,
                                    size_t rank_offset) const
    {
        auto axes = parse_vector_attribute(info, name, expected_size);
        if(axes.empty())
            return std::vector<int64_t>(expected_size, 0);

        std::transform(axes.begin(),
                       axes.end(),
                       ins_begin,
                       axes.begin(),
                       [&](int64_t axis, instruction_ref arg) {
                           return normalize_axis(axis, arg->get_shape().ndim() + rank_offset);
                       });

        return axes;
    }

    void modify_body(module_ref mod,
                     const std::vector<instruction_ref>& args,
                     int64_t M,
                     int64_t N,
                     const std::vector<int64_t>& scan_input_axes,
                     const std::vector<int64_t>& scan_input_directions) const
    {
        auto param_names  = mod->get_parameter_names();
        auto param_shapes = mod->get_parameter_shapes();

        std::unordered_map<std::string, std::vector<instruction_ref>> child_ins;
        for(auto ins : iterator_for(*mod))
        {
            for(const auto& name : param_names)
            {
                auto param = mod->get_parameter(name);
                if(contains(ins->inputs(), param))
                    child_ins[name].push_back(ins);
            }
        }

        auto iter_param = mod->add_parameter("iter", shape{shape::int64_type});
        auto cond_param = mod->add_parameter("cond", shape{shape::bool_type});
        for(auto i = 0; i < M; ++i)
        {
            auto var =
                mod->add_parameter("state_var" + std::to_string(i), param_shapes[param_names[i]]);
            auto param = mod->get_parameter(param_names[i]);
            for(auto ins : child_ins[param_names[i]])
                ins->replace_argument(ins, param, var);
            mod->remove_instruction(param);
        }

        std::vector<instruction_ref> scan_in_params;
        scan_in_params.reserve(N);
        for(auto i = M; i < M + N; ++i)
        {
            auto param = mod->get_parameter(param_names[i]);
            auto scan_in_param =
                mod->add_parameter("scan_in" + std::to_string(i - M), args[i]->get_shape());
            scan_in_params.push_back(scan_in_param);
            auto scan_axis     = scan_input_axes[i - M];
            auto scan_dir      = scan_input_directions[i - M];
            auto scan_in_slice = mod->insert_instruction(
                param,
                make_op("scan_slice", {{"axis", scan_axis}, {"direction", scan_dir}}),
                scan_in_param,
                iter_param);
            scan_in_slice = mod->insert_instruction(
                param, make_op("squeeze", {{"axes", {scan_axis}}}), scan_in_slice);
            for(auto ins : child_ins[param_names[i]])
                ins->replace_argument(ins, param, scan_in_slice);
            mod->remove_instruction(param);
        }
        auto returns = mod->get_returns();
        returns.insert(returns.begin(), cond_param);
        returns.insert(returns.begin() + M + 1, scan_in_params.begin(), scan_in_params.end());
        mod->replace_return(returns);
    }
}; // namespace onnx

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
