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
#include <migraphx/functional.hpp>
#include <algorithm>
#include <numeric>
#include <optional>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// The extent dyn_slice gives for a bound on an axis of length len
static int64_t clipped_bound(int64_t bound, int64_t len)
{
    if(bound < 0)
        bound += len;
    return std::clamp<int64_t>(bound, 0, len);
}

// The comparisons in `e` whose operands have none
static std::vector<sym::expr> innermost_comparisons(const sym::expr& e)
{
    std::vector<sym::expr> result;
    fix<bool>([&](auto self, const sym::expr& x) {
        bool nested = false;
        for(const auto& child : x.children())
            nested = self(child) or nested;
        const bool comparison = contains({"less_than", "equal_to"}, std::string{x.name()});
        if(comparison and not nested and not contains(result, x))
            result.push_back(x);
        return comparison or nested;
    })(e);
    return result;
}

// Whether `len` is the only symbol in `e`
static bool depends_only_on(const sym::expr& len, const sym::expr& e)
{
    const auto allowed = sym::find_variables(len);
    const auto vars    = sym::find_variables(e);
    return std::all_of(
        vars.begin(), vars.end(), [&](const auto& v) { return contains(allowed, v); });
}

// The symbolic value of a bound, with a comparison of the axis length resolved: a bound such as
// `n > 100 ? 100 : -1` for a clip to 100, computed with arithmetic as tf2onnx exports TF's
// pad_or_clip, becomes an expression of the length when one candidate gives the same extent as the
// bound for every length the axis can have
static std::optional<sym::expr> resolve_bound(const sym::expr& bound,
                                              const shape::dynamic_dimension& dim)
{
    const auto comparisons = innermost_comparisons(bound);
    if(comparisons.empty())
        return bound;
    if(comparisons.size() != 1 or not dim.is_symbolic() or dim.sym_expr.name() != "variable")
        return std::nullopt;
    // Only the axis length may vary, so checking each of its values covers every case
    const auto& len   = dim.sym_expr;
    const auto bounds = dim.get_interval();
    if(bounds.max - bounds.min > (1u << 16) or not depends_only_on(len, bound))
        return std::nullopt;

    std::vector<sym::expr> branches;
    for(int64_t outcome : {0, 1})
    {
        auto branch = bound.subs({{comparisons.front(), sym::lit(outcome)}});
        if(not innermost_comparisons(branch).empty())
            return std::nullopt;
        branches.push_back(branch);
    }
    std::vector<sym::expr> candidates = {
        sym::min(len, branches[0]), sym::min(len, branches[1]), branches[0], branches[1]};
    auto matches = [&](const sym::expr& candidate) {
        const int64_t lo = bounds.min;
        const int64_t hi = bounds.max;
        for(auto n = lo; n <= hi; ++n)
        {
            std::unordered_map<sym::expr, sym::scalar> vars = {{len, sym::scalar{n}}};
            if(clipped_bound(sym::to<int64_t>(candidate.eval(vars)), n) !=
               clipped_bound(sym::to<int64_t>(bound.eval(vars)), n))
                return false;
        }
        return true;
    };
    auto found = std::find_if(candidates.begin(), candidates.end(), matches);
    if(found == candidates.end())
        return std::nullopt;
    return *found;
}

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
            // Each bound is its symbolic value, or a new symbol bounded by the axis length where it
            // is only known at run time or depends on a comparison that does not resolve, since
            // dyn_slice clips it to the axis
            if(axes.has_value() and steps.has_value() and steps->size() == axes->size() and
               all_of(*steps, [](auto step) { return step == 1; }) and
               args[1]->get_shape().elements() == axes->size() and
               args[2]->get_shape().elements() == axes->size())
            {
                const auto& dims = args[0]->get_shape().dyn_dims();
                auto bounds      = [&](const sym_argument& value, const std::string& suffix) {
                    const auto values =
                        value.empty() ? std::vector<sym::expr>{} : value.get().to_vector();
                    std::vector<sym::expr> result;
                    for(std::size_t i = 0; i < axes->size(); ++i)
                    {
                        auto axis       = axes->at(i);
                        const auto& dim = dims.at(axis < 0 ? axis + dims.size() : axis);
                        std::optional<sym::expr> bound;
                        if(i < values.size())
                            bound = resolve_bound(values[i], dim);
                        if(not bound.has_value())
                            bound = sym::var(parser.symbol_names.allocate(info.name + suffix),
                                             {0, static_cast<int64_t>(dim.get_interval().max)});
                        result.push_back(*bound);
                    }
                    return result;
                };
                return info.add_instruction(make_op("dyn_slice",
                                                    {{"axes", *axes},
                                                     {"starts", to_value(bounds(starts, "_start"))},
                                                     {"ends", to_value(bounds(ends, "_end"))}}),
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
