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
#include <migraphx/op/convert.hpp>
#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Evaluates one element of an integer tensor from literals and symbolic values, as for the slice
// bounds TF's pad_or_clip computes from the input shape, when the tensor as a whole has no
// symbolic value. A comparison of values that are not fixed takes the outcome in `decisions`, or
// is recorded in `open` and stops the evaluation.
struct bound_evaluator
{
    std::unordered_map<instruction_ref, bool> decisions = {};
    std::vector<instruction_ref> open                   = {};

    std::optional<sym::expr> operator()(instruction_ref ins, std::size_t i, int depth = 0)
    {
        const auto& s = ins->get_shape();
        if(depth > 64 or s.dynamic() or s.elements() == 0)
            return std::nullopt;
        if(s.elements() == 1)
            i = 0;
        else if(i >= s.elements())
            return std::nullopt;
        const auto value = ins->sym_eval();
        if(not value.empty())
        {
            const auto values = value.get().to_vector();
            if(i < values.size())
                return values[i];
        }
        auto input = [&](std::size_t k, std::size_t j) -> std::optional<sym::expr> {
            auto x = ins->inputs().at(k);
            auto n = x->get_shape().elements();
            if(n != 1 and n != s.elements())
                return std::nullopt;
            return (*this)(x, j, depth + 1);
        };
        const auto& name = ins->name();
        if(contains({"multibroadcast", "unsqueeze", "squeeze", "reshape", "contiguous"}, name))
            return input(0, i);
        if(name == "convert")
        {
            auto x = input(0, i);
            if(not x.has_value() or
               any_cast<op::convert>(ins->get_operator()).target_type != shape::bool_type)
                return x;
            auto v = sym::fixed_value(*x);
            if(not v.has_value())
                return std::nullopt;
            return sym::lit(sym::to<double>(*v) != 0 ? 1 : 0);
        }
        if(name == "not")
        {
            auto x = input(0, i);
            auto v = x.has_value() ? sym::fixed_value(*x) : std::nullopt;
            if(not v.has_value())
                return std::nullopt;
            return sym::lit(sym::to<double>(*v) == 0 ? 1 : 0);
        }
        if(name == "gather" and ins->inputs().front()->get_shape().ndim() == 1)
        {
            auto index = (*this)(ins->inputs().at(1), i, depth + 1);
            auto value = index.has_value() ? sym::fixed_value(*index) : std::nullopt;
            if(not value.has_value())
                return std::nullopt;
            auto data = ins->inputs().front();
            auto j    = sym::to<int64_t>(*value);
            if(j < 0)
                j += data->get_shape().elements();
            return (*this)(data, j, depth + 1);
        }
        if(name == "slice" and ins->inputs().size() == 1 and
           ins->inputs().front()->get_shape().ndim() == 1)
        {
            auto data  = ins->inputs().front();
            auto op    = any_cast<op::slice>(ins->get_operator());
            auto start = op.starts.empty() ? 0 : op.starts.front();
            if(start < 0)
                start += data->get_shape().elements();
            return (*this)(data, i + start, depth + 1);
        }
        if(name == "concat" and s.ndim() == 1)
        {
            for(auto x : ins->inputs())
            {
                auto n = x->get_shape().elements();
                if(i < n)
                    return (*this)(x, i, depth + 1);
                i -= n;
            }
            return std::nullopt;
        }
        if(contains({"add", "sub", "mul"}, name))
        {
            auto a = input(0, i);
            auto b = input(1, i);
            if(not a.has_value() or not b.has_value())
                return std::nullopt;
            if(name == "add")
                return *a + *b;
            if(name == "sub")
                return *a - *b;
            return *a * *b;
        }
        if(contains({"greater", "less", "equal"}, name))
        {
            auto a = input(0, i);
            auto b = input(1, i);
            if(not a.has_value() or not b.has_value())
                return std::nullopt;
            auto va = sym::fixed_value(*a);
            auto vb = sym::fixed_value(*b);
            if(va.has_value() and vb.has_value())
                return sym::lit(compare(name, sym::to<double>(*va), sym::to<double>(*vb)) ? 1 : 0);
            if(s.elements() != 1)
                return std::nullopt;
            auto decision = decisions.find(ins);
            if(decision != decisions.end())
                return sym::lit(decision->second ? 1 : 0);
            if(not contains(open, ins))
                open.push_back(ins);
            return std::nullopt;
        }
        return std::nullopt;
    }

    static bool compare(const std::string& name, double a, double b)
    {
        if(name == "greater")
            return a > b;
        if(name == "less")
            return a < b;
        return a == b;
    }
};

// The extent dyn_slice gives for an end on an axis of length len
static int64_t clipped_end(int64_t end, int64_t len)
{
    if(end < 0)
        end += len;
    return std::clamp<int64_t>(end, 0, len);
}

// An end that depends on one comparison of the axis length, such as `n > 100 ? 100 : -1` for a
// clip to 100, resolves to an expression of the length when one candidate gives the same extent as
// both cases for every length the axis can have
static std::optional<sym::expr>
resolve_conditional_end(instruction_ref ends, std::size_t i, const shape::dynamic_dimension& dim)
{
    bound_evaluator probe;
    if(auto end = probe(ends, i))
        return end;
    if(probe.open.size() != 1 or not dim.is_symbolic() or dim.sym_expr.name() != "variable")
        return std::nullopt;
    auto condition   = probe.open.front();
    const auto range = dim.get_interval();
    if(range.max - range.min > (1u << 16))
        return std::nullopt;

    std::array<sym::expr, 2> branches;
    for(bool outcome : {false, true})
    {
        bound_evaluator evaluator{{{condition, outcome}}};
        auto end = evaluator(ends, i);
        if(not end.has_value() or not evaluator.open.empty())
            return std::nullopt;
        branches[outcome] = *end;
    }
    bound_evaluator operands;
    auto lhs = operands(condition->inputs().at(0), 0);
    auto rhs = operands(condition->inputs().at(1), 0);
    if(not lhs.has_value() or not rhs.has_value())
        return std::nullopt;

    const auto& len                   = dim.sym_expr;
    std::vector<sym::expr> candidates = {
        sym::min(len, branches[0]), sym::min(len, branches[1]), branches[0], branches[1]};
    auto matches = [&](const sym::expr& candidate) {
        for(auto n = static_cast<int64_t>(range.min); n <= static_cast<int64_t>(range.max); ++n)
        {
            std::unordered_map<sym::expr, sym::scalar> vars = {{len, sym::scalar{n}}};
            auto outcome  = bound_evaluator::compare(condition->name(),
                                                     sym::to<double>(lhs->eval(vars)),
                                                     sym::to<double>(rhs->eval(vars)));
            auto expected = clipped_end(sym::to<int64_t>(branches[outcome].eval(vars)), n);
            if(clipped_end(sym::to<int64_t>(candidate.eval(vars)), n) != expected)
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
            // A start or end only known at run time gets a symbol per axis, bounded by the axis
            // length, since dyn_slice clips it to the axis
            if((starts.empty() or ends.empty()) and axes.has_value() and steps.has_value() and
               steps->size() == axes->size() and
               all_of(*steps, [](auto step) { return step == 1; }) and
               args[1]->get_shape().elements() == axes->size() and
               args[2]->get_shape().elements() == axes->size())
            {
                const auto& dims = args[0]->get_shape().dyn_dims();
                // Each bound is worked out per axis where it can be; the others get a symbol
                // bounded by the axis length
                auto bounds = [&](const sym_argument& known,
                                  instruction_ref arg,
                                  const std::string& suffix,
                                  bool is_end) {
                    if(not known.empty())
                        return known.get().to_vector();
                    std::vector<sym::expr> result;
                    for(std::size_t i = 0; i < axes->size(); ++i)
                    {
                        auto axis       = axes->at(i);
                        const auto& dim = dims.at(axis < 0 ? axis + dims.size() : axis);
                        auto bound      = is_end ? resolve_conditional_end(arg, i, dim)
                                                 : bound_evaluator{}(arg, i);
                        if(not bound.has_value())
                            bound = sym::var(parser.symbol_names.allocate(info.name + suffix),
                                             {0, static_cast<int64_t>(dim.get_interval().max)});
                        result.push_back(*bound);
                    }
                    return result;
                };
                return info.add_instruction(
                    make_op("dyn_slice",
                            {{"axes", *axes},
                             {"starts", to_value(bounds(starts, args[1], "_start", false))},
                             {"ends", to_value(bounds(ends, args[2], "_end", true))}}),
                    args[0],
                    args[1],
                    args[2]);
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
