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
        parser.parse_graph(body, body_graph);

        auto body_outs = body->get_returns();
        const auto M   = info.attributes["num_scan_inputs"].i();
        const auto N   = args.size() - M;
        const auto K   = body_outs.size() - N;

        if(body->get_parameter_names().size() != N + M)
            MIGRAPHX_THROW("Lorem ipsum");

        const auto scan_input_axes = parse_axes(info, "scan_input_axes", M, args.begin() + N, 0);

        size_t num_iters = args[N]->get_shape().lens()[scan_input_axes[0]];
        for(auto i = 1; i < M; ++i)
            if(args[i]->get_shape().lens()[scan_input_axes[i]] != num_iters)
                MIGRAPHX_THROW("Lorem ipsum");

        const auto scan_input_directions = parse_dirs(info, "scan_input_directions", M);

        const auto scan_output_axes =
            parse_axes(info, "scan_output_axes", K, body_outs.begin() + N, 1);

        const auto scan_output_directions = parse_dirs(info, "scan_output_directions", K);

        // TODO check that alt_args shapes match body input parameter shapes

        modify_body(body, args, N, M, scan_input_axes, scan_input_directions);

        auto max_iter_lit = info.add_literal(literal{shape{shape::int64_type}, {num_iters}});
        auto cond_lit     = info.add_literal(literal{shape{shape::bool_type}, {true}});
        std::vector<instruction_ref> loop_args{max_iter_lit, cond_lit};
        loop_args.insert(loop_args.end(), args.begin(), args.begin() + N);

        auto loop = info.add_instruction(
            make_op("loop", {{"max_iterations", num_iters}}), loop_args, {body});

        std::vector<instruction_ref> ret;
        ret.reserve(N + K);
        for(auto i = 0; i < N; ++i)
            ret.push_back(info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), loop));

        for(auto i = 0; i < K; ++i)
        {
            auto o    = info.add_instruction(make_op("get_tuple_elem", {{"index", i + N}}), loop);
            auto perm = make_perm_for_scan_out(o->get_shape().ndim(), scan_output_axes[i]);
            ret.push_back(info.add_instruction(make_op("transpose", {{"permutation", perm}}), o));
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
                     int64_t N,
                     int64_t M,
                     const std::vector<int64_t>& scan_input_axes,
                     const std::vector<int64_t>& scan_input_directions) const
    {
        std::vector<instruction_ref> params;
        params.reserve(N + M);
        transform(mod->get_parameter_names(),
                  std::back_inserter(params),
                  [&](const std::string& name) { return mod->get_parameter(name); });

        auto iter_param = mod->add_parameter("iter", shape{shape::int64_type});
        auto cond_param = mod->add_parameter("cond", shape{shape::bool_type});
        std::vector<instruction_ref> new_params;
        new_params.reserve(N);
        for(auto i = 0; i < N; ++i)
            new_params.push_back(
                mod->add_parameter("state_var" + std::to_string(i), params[i]->get_shape()));

        for(auto i = 0; i < params.size(); ++i)
        {
            for(auto ins : iterator_for(*mod))
            {
                if(not contains(ins->inputs(), params[i]))
                    continue;

                auto new_ins = i < N ? new_params[i] : args[i];
                if(i >= N)
                {
                    auto scan_axis = scan_input_axes[i - N];
                    auto scan_dir  = scan_input_directions[i - N];
                    new_ins        = mod->insert_instruction(
                        params[i],
                        make_op("scan_slice", {{"axis", scan_axis}, {"direction", scan_dir}}),
                        new_ins,
                        iter_param);
                    new_ins = mod->insert_instruction(
                        params[i], make_op("squeeze", {{"axes", {scan_axis}}}), new_ins);
                }
                ins->replace_argument(ins, params[i], new_ins);
            }
            mod->remove_instruction(params[i]);
        }

        auto returns = mod->get_returns();
        returns.insert(returns.begin(), cond_param);
        mod->replace_return(returns);
    }

    std::vector<int64_t> make_perm_for_scan_out(int64_t rank, int64_t axis) const
    {
        std::vector<int64_t> perm(rank);
        std::iota(perm.begin(), perm.end(), 0);
        std::copy(perm.begin() + 1, perm.begin() + 1 + axis, perm.begin());
        perm[axis] = 0;

        return perm;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
