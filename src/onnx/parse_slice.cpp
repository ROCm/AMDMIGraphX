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
#include <migraphx/sym.hpp>
#include <migraphx/tune_axis.hpp>
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

    /**
     * Slice a static or symbolic input by run-time starts and/or ends with dyn_slice. Each run-time
     * bound becomes a symbol bounded by its axis length, so the result has a symbolic shape rather
     * than a range-based one; ops such as TopK with a run-time k need a symbolic input. Returns
     * nullopt when this does not apply: constant bounds, run-time axes, or non-unit steps.
     */
    static std::optional<instruction_ref>
    parse_runtime_bounds(onnx_parser& parser,
                         const onnx_parser::node_info& info,
                         const std::vector<instruction_ref>& args)
    {
        const auto& data_shape = args[0]->get_shape();
        if(args.size() < 3 or (data_shape.dynamic() and not data_shape.symbolic()))
            return std::nullopt;
        const auto starts_arg = args[1]->eval();
        const auto ends_arg   = args[2]->eval();
        if(not starts_arg.empty() and not ends_arg.empty())
            return std::nullopt;
        const auto& starts_shape = args[1]->get_shape();
        const auto& ends_shape   = args[2]->get_shape();
        if(starts_shape.dynamic() or ends_shape.dynamic() or starts_shape.ndim() != 1 or
           starts_shape.lens() != ends_shape.lens())
            return std::nullopt;
        const auto n = starts_shape.elements();

        std::vector<int64_t> axes(n);
        if(args.size() >= 4)
        {
            const auto axes_arg = args[3]->eval();
            if(axes_arg.empty())
                return std::nullopt;
            axes = axes_arg.to_vector<int64_t>();
        }
        else if(contains(info.attributes, "axes"))
        {
            axes.clear();
            literal s = parser.parse_value(info.attributes.at("axes"));
            s.visit([&](auto v) { copy(v, std::back_inserter(axes)); });
        }
        else
        {
            std::iota(axes.begin(), axes.end(), int64_t{0});
        }
        if(axes.size() != n)
            return std::nullopt;

        if(args.size() >= 5)
        {
            const auto steps_arg = args[4]->eval();
            if(steps_arg.empty())
                return std::nullopt;
            const auto steps = steps_arg.to_vector<int64_t>();
            if(std::any_of(steps.begin(), steps.end(), [](auto s) { return s != 1; }))
                return std::nullopt;
        }

        auto bounds = [&](const argument& arg, const std::string& kind) {
            std::vector<int64_t> values;
            if(not arg.empty())
                values = arg.to_vector<int64_t>();
            std::vector<sym::expr> result;
            for(std::size_t i = 0; i < n; ++i)
            {
                if(not values.empty())
                {
                    result.push_back(sym::lit(values[i]));
                    continue;
                }
                auto axis = tune_axis(data_shape.ndim(), axes[i], "PARSE_SLICE");
                auto len  = static_cast<int64_t>(data_shape.max_lens().at(axis));
                result.push_back(sym::var(
                    parser.symbol_names.allocate(info.name + "_" + kind + std::to_string(i)),
                    {0, len}));
            }
            return result;
        };
        return info.add_instruction(make_op("dyn_slice",
                                            {{"axes", axes},
                                             {"starts", to_value(bounds(starts_arg, "start"))},
                                             {"ends", to_value(bounds(ends_arg, "end"))}}),
                                    args[0],
                                    args[1],
                                    args[2]);
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() >= 3 and args[0]->get_shape().symbolic())
        {
            const auto starts = args[1]->sym_eval();
            const auto ends   = args[2]->sym_eval();
            std::optional<std::vector<int64_t>> axes;
            if(args.size() >= 4)
            {
                const auto axes_value = args[3]->sym_eval();
                if(not axes_value.empty())
                    axes = sym::fixed_values<int64_t>(axes_value.get());
            }
            else if(not starts.empty())
            {
                axes.emplace(starts.get_shape().elements());
                std::iota(axes->begin(), axes->end(), int64_t{0});
            }

            std::optional<std::vector<int64_t>> steps;
            if(args.size() >= 5)
            {
                const auto steps_value = args[4]->sym_eval();
                if(not steps_value.empty())
                    steps = sym::fixed_values<int64_t>(steps_value.get());
            }
            else if(axes.has_value())
            {
                steps.emplace(axes->size(), int64_t{1});
            }
            if(not starts.empty() and not ends.empty() and axes.has_value() and
               steps.has_value() and starts.get_shape().elements() == axes->size() and
               ends.get_shape().elements() == axes->size() and steps->size() == axes->size() and
               all_of(*steps, [](auto step) { return step == 1; }))
            {
                const auto starts_values = starts.get().to_vector();
                const auto ends_values   = ends.get().to_vector();
                return info.add_instruction(make_op("dyn_slice",
                                                    {{"axes", *axes},
                                                     {"starts", to_value(starts_values)},
                                                     {"ends", to_value(ends_values)}}),
                                            args[0],
                                            args[1],
                                            args[2]);
            }
        }

        if(auto ins = parse_runtime_bounds(parser, info, args))
            return *ins;

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
