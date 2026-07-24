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
#include <migraphx/op/slice.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dim_like.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_slice : op_parser<parse_slice>
{

    std::vector<op_desc> operators() const { return {{"Slice"}}; }

    struct slice_desc
    {
        std::vector<instruction_ref> op_args;
        std::vector<int64_t> axes;
        std::vector<dim_like> starts;
        std::vector<dim_like> ends;
        std::vector<int64_t> steps;
        std::vector<int64_t> raxes;
        std::vector<op::slice::slice_mode> mode{};

        void always_insert(instruction_ref arg) { op_args.insert(op_args.begin(), arg); }

        // Either insert argument into `this->op_args` or return the constant value of the argument
        std::vector<int64_t> try_insert(instruction_ref arg)
        {
            std::vector<int64_t> result;
            migraphx::argument arg_value = arg->eval();
            if(arg_value.empty())
            {
                op_args.insert(op_args.begin(), arg);
            }
            else
            {
                arg_value.visit([&](auto s) { result.assign(s.begin(), s.end()); });
            }
            return result;
        }

        op::slice create_slice_operator() { return op::slice{axes, starts, ends, mode}; }
    };

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        auto sd  = construct_slice_desc(parser, info, args);
        auto ins = info.add_instruction(sd.create_slice_operator(), sd.op_args);
        if(not sd.raxes.empty())
        {
            ins = info.add_instruction(make_op("reverse", {{"axes", sd.raxes}}), ins);
        }
        // If any steps are other than default 1, add a "steps" op
        if(std::any_of(sd.steps.begin(), sd.steps.end(), [](auto s) { return std::abs(s) != 1; }))
        {
            std::vector<int64_t> nsteps;
            std::transform(sd.steps.begin(),
                           sd.steps.end(),
                           std::back_inserter(nsteps),
                           [](auto s) { return std::abs(s); });
            return ins = info.add_instruction(
                       make_op("step", {{"axes", sd.axes}, {"steps", nsteps}}), ins);
        }
        else
            return ins;
    }

    slice_desc handle_sd_inputs(const onnx_parser& parser, onnx_parser::node_info info, std::vector<instruction_ref> args) const
    {
        slice_desc sd;
        // ONNX Slice can have up to 5 inputs, we first check the 5th one
        // to decide whether MIGX can handle this slice.
        if(args.size() == 5)
        {
            migraphx::argument step_arg = args.back()->eval();
            check_arg_empty(step_arg, "PARSE_SLICE: cannot handle variable steps for slice");
            step_arg.visit([&](auto s) { sd.steps.assign(s.begin(), s.end()); });
        }
        if(args.size() >= 4)
        {
            auto tmp_axes = sd.try_insert(args.at(3));
            if(tmp_axes.empty())
                sd.mode.insert(sd.mode.begin(), op::slice::slice_mode::axes);
            sd.axes = tmp_axes;
        }
        else if(contains(info.attributes, "axes"))
        {
            literal s = parser.parse_value(info.attributes.at("axes"));
            s.visit([&](auto v) { copy(v, std::back_inserter(sd.axes)); });
        }
        // NOTE: goes through range-based dynamic shapes pathway only
        if(args.size() >= 3)
        {
            auto tmp_ends = sd.try_insert(args.at(2));
            if(tmp_ends.empty())
                sd.mode.insert(sd.mode.begin(), op::slice::slice_mode::ends);
            sd.ends.assign(tmp_ends.begin(), tmp_ends.end());
        }
        else if(contains(info.attributes, "ends"))
        {
            literal s = parser.parse_value(info.attributes.at("ends"));
            s.visit([&](auto v) {
                std::transform(v.begin(), v.end(), std::back_inserter(sd.ends), [](auto e) {
                    return static_cast<int64_t>(e);
                });
            });
        }
        if(args.size() >= 2)
        {
            auto tmp_starts = sd.try_insert(args.at(1));
            if(tmp_starts.empty())
                sd.mode.insert(sd.mode.begin(), op::slice::slice_mode::starts);
            sd.starts.assign(tmp_starts.begin(), tmp_starts.end());
        }
        else if(contains(info.attributes, "starts"))
        {
            literal s = parser.parse_value(info.attributes.at("starts"));
            s.visit([&](auto v) {
                std::transform(v.begin(), v.end(), std::back_inserter(sd.starts), [](auto e) {
                    return static_cast<int64_t>(e);
                });
            });
        }
        // data input argument
        sd.always_insert(args.at(0));
        return sd;
    }

    slice_desc construct_slice_desc(const onnx_parser& parser,
                                    onnx_parser::node_info info,
                                    std::vector<instruction_ref> args) const
    {
        slice_desc sd = handle_sd_inputs(parser, info, args);
        // If axes arg is not given, the default is all of them.
        if(sd.axes.empty() and sd.op_args.size() <= 3)
        {
            std::vector<int64_t> axes(args[0]->get_shape().ndim());
            std::iota(axes.begin(), axes.end(), int64_t{0});
            sd.axes = axes;
        }

        if(std::any_of(sd.steps.begin(), sd.steps.end(), [](auto s) { return s != 1; }))
        {
            if(sd.starts.empty() or sd.ends.empty())
                MIGRAPHX_THROW(
                    "PARSE_SLICE: steps and variable starts and/or ends is not supported");
            if(sd.axes.empty())
                MIGRAPHX_THROW("PARSE_SLICE: steps and variable axes is not supported");
        }

        // If any axes have negative step, prepare to add a "reverse" op
        for(auto i : range(sd.steps.size()))
        {
            if(sd.steps[i] >= 0)
                continue;
            auto start = std::get<int64_t>(sd.starts[i]) + 1;
            if(start == 0)
                start = INT_MAX;
            sd.starts[i] = start;
            sd.ends[i]   = std::get<int64_t>(sd.ends[i]) + 1;
            sd.raxes.push_back(sd.axes[i]);
            std::swap(sd.starts[i], sd.ends[i]);
        }
        return sd;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
