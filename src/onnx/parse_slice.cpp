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
#include <numeric>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_slice : op_parser<parse_slice>
{

    std::vector<op_desc> operators() const { return {{"Slice"}}; }

    struct slice_desc
    {
        op::slice op;
        std::vector<instruction_ref> op_args;
        std::vector<int64_t> steps;
        std::vector<int64_t> raxes;

        void always_insert(instruction_ref arg) { op_args.insert(op_args.begin(), arg); }

        /**
         * Either insert argument into `this->op_args` or return the constant value of the argument
         */
        std::vector<int64_t> insert(instruction_ref arg)
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
    };

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() >= 3 and args[0]->get_shape().symbolic())
        {
            const auto starts = parser.get_symbolic_tensor_value(args[1]);
            const auto ends   = parser.get_symbolic_tensor_value(args[2]);
            std::optional<std::vector<int64_t>> axes;
            if(args.size() >= 4)
            {
                const auto axes_value = parser.get_symbolic_tensor_value(args[3]);
                if(axes_value.has_value())
                    axes = fixed_integers(*axes_value);
            }
            else if(starts.has_value())
            {
                axes.emplace(starts->size());
                std::iota(axes->begin(), axes->end(), int64_t{0});
            }

            std::optional<std::vector<int64_t>> steps;
            if(args.size() >= 5)
            {
                const auto steps_value = parser.get_symbolic_tensor_value(args[4]);
                if(steps_value.has_value())
                    steps = fixed_integers(*steps_value);
            }
            else if(axes.has_value())
            {
                steps.emplace(axes->size(), int64_t{1});
            }
            if(starts.has_value() and ends.has_value() and axes.has_value() and
               steps.has_value() and starts->size() == axes->size() and
               ends->size() == axes->size() and steps->size() == axes->size() and
               all_of(*steps, [](auto step) { return step == 1; }))
            {
                return info.add_instruction(make_op("dyn_slice",
                                                    {{"axes", *axes},
                                                     {"starts", to_value(*starts)},
                                                     {"ends", to_value(*ends)}}),
                                            args[0],
                                            args[1],
                                            args[2]);
            }
        }

        auto sd  = construct_slice_desc(parser, info, args);
        auto ins = info.add_instruction(sd.op, sd.op_args);
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
            return info.add_instruction(make_op("step", {{"axes", sd.op.axes}, {"steps", nsteps}}),
                                        ins);
        }
        else
            return ins;
    }

    void infer_symbolic_values(const op_desc&, const symbolic_propagate_context& context) const
    {
        const auto data = context.arg(0);
        if(not data.has_value() or context.args[0]->get_shape().ndim() != 1 or
           context.args.size() < 3)
            return;
        const auto starts = context.arg(1);
        const auto ends   = context.arg(2);
        const auto axes   = context.args.size() > 3
                                ? context.arg(3)
                                : std::make_optional(symbolic_tensor_value{sym::lit(int64_t{0})});
        const auto steps  = context.args.size() > 4
                                ? context.arg(4)
                                : std::make_optional(symbolic_tensor_value{sym::lit(int64_t{1})});
        if(not starts.has_value() or not ends.has_value() or not axes.has_value() or
           not steps.has_value())
            return;

        const auto fixed_starts = fixed_integers(*starts);
        const auto fixed_ends   = fixed_integers(*ends);
        const auto fixed_axes   = fixed_integers(*axes);
        const auto fixed_steps  = fixed_integers(*steps);
        if(not fixed_starts.has_value() or not fixed_ends.has_value() or
           not fixed_axes.has_value() or not fixed_steps.has_value() or fixed_starts->size() != 1 or
           fixed_ends->size() != 1 or fixed_axes->size() != 1 or fixed_steps->size() != 1 or
           fixed_axes->front() != 0 or fixed_steps->front() != 1)
            return;

        const auto size      = static_cast<int64_t>(data->size());
        const auto normalize = [size](int64_t index) {
            if(index < 0)
                index += size;
            return std::clamp(index, int64_t{0}, size);
        };
        const auto start = normalize(fixed_starts->front());
        const auto end   = normalize(fixed_ends->front());
        symbolic_tensor_value result{data->begin() + start, data->begin() + std::max(start, end)};
        if(context.output_has_elements(result.size()))
            context.set(std::move(result));
    }

    slice_desc construct_slice_desc(const onnx_parser& parser,
                                    onnx_parser::node_info info,
                                    std::vector<instruction_ref> args) const
    {
        slice_desc sd;

        // slice can have up to 5 inputs, we first check the 5th one
        // to decide whether MIGRAPHX can handle this slice.
        if(args.size() == 5)
        {
            migraphx::argument step_arg = args.back()->eval();
            check_arg_empty(step_arg, "PARSE_SLICE: cannot handle variable steps for slice");
            step_arg.visit([&](auto s) { sd.steps.assign(s.begin(), s.end()); });
        }

        if(args.size() >= 4)
        {
            sd.op.axes = sd.insert(args.at(3));
        }
        else if(contains(info.attributes, "axes"))
        {
            literal s = parser.parse_value(info.attributes.at("axes"));
            s.visit([&](auto v) { copy(v, std::back_inserter(sd.op.axes)); });
        }

        if(args.size() >= 3)
        {
            sd.op.ends = sd.insert(args.at(2));
        }
        else if(contains(info.attributes, "ends"))
        {
            literal s = parser.parse_value(info.attributes.at("ends"));
            s.visit([&](auto v) { copy(v, std::back_inserter(sd.op.ends)); });
        }

        if(args.size() >= 2)
        {
            sd.op.starts = sd.insert(args.at(1));
        }
        else if(contains(info.attributes, "starts"))
        {
            literal s = parser.parse_value(info.attributes.at("starts"));
            s.visit([&](auto v) { copy(v, std::back_inserter(sd.op.starts)); });
        }

        // data input argument
        sd.always_insert(args.at(0));

        // If axes arg is not given, the default is all of them.
        if(sd.op.axes.empty() and sd.op_args.size() <= 3)
        {
            std::vector<int64_t> axes(args[0]->get_shape().ndim());
            std::iota(axes.begin(), axes.end(), int64_t{0});
            sd.op.axes = axes;
        }

        if(std::any_of(sd.steps.begin(), sd.steps.end(), [](auto s) { return s != 1; }))
        {
            if(sd.op.starts.empty() or sd.op.ends.empty())
                MIGRAPHX_THROW(
                    "PARSE_SLICE: steps and variable starts and/or ends is not supported");
            if(sd.op.axes.empty())
                MIGRAPHX_THROW("PARSE_SLICE: steps and variable axes is not supported");
        }

        // If any axes have negative step, prepare to add a "reverse" op
        for(auto i : range(sd.steps.size()))
        {
            if(sd.steps[i] >= 0)
                continue;
            sd.op.starts[i] += 1;
            if(sd.op.starts[i] == 0)
                sd.op.starts[i] = INT_MAX;
            sd.op.ends[i] += 1;
            sd.raxes.push_back(sd.op.axes[i]);
            std::swap(sd.op.starts[i], sd.op.ends[i]);
        }
        return sd;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
