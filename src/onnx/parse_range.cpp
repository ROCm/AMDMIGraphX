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
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/sym_argument.hpp>
#include <cmath>
#include <numeric>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// The range of values an instruction can hold, inferred from the ranges of its inputs, or nullopt
// when it cannot be bounded. It bounds a value only known at run time, such as a count of selected
// elements, so the length that depends on it can be a bounded symbol.
static std::optional<sym::interval> value_bounds(instruction_ref ins, int depth = 0)
{
    const auto bool_range = sym::interval{int64_t{0}, int64_t{1}};
    auto merge            = [](const sym::interval& a, const sym::interval& b) {
        return sym::interval{sym::scalar_min(a.min, b.min), sym::scalar_max(a.max, b.max)};
    };
    const auto& s = ins->get_shape();
    if((s.dynamic() and not s.symbolic()) or s.type() == shape::tuple_type)
        return std::nullopt;
    const auto value = ins->sym_eval();
    if(not value.empty())
    {
        std::optional<sym::interval> result;
        for(const auto& e : value.get())
        {
            auto i = e.eval_interval();
            result = result.has_value() ? merge(*result, i) : i;
        }
        return result;
    }
    if(depth > 64 or ins->inputs().empty())
        return std::nullopt;
    const auto& name = ins->name();
    auto input       = [&](std::size_t i) { return value_bounds(ins->inputs().at(i), depth + 1); };
    if(contains({"equal", "less", "greater", "not", "logical_and", "logical_or", "logical_xor"},
                name))
        return bool_range;
    if(name == "convert")
    {
        if(ins->get_shape().type() == shape::bool_type)
            return bool_range;
        auto x = input(0);
        if(not x.has_value() or not shape::is_integral(ins->get_shape().type()))
            return x;
        // A float truncates toward zero, so it stays within the floor and ceil of its range
        return sym::interval{static_cast<int64_t>(std::floor(sym::to<double>(x->min))),
                             static_cast<int64_t>(std::ceil(sym::to<double>(x->max)))};
    }
    // Ops that only move or repeat the values of their first input
    if(contains({"identity",
                 "contiguous",
                 "squeeze",
                 "unsqueeze",
                 "reshape",
                 "reshape_lazy",
                 "flatten",
                 "transpose",
                 "slice",
                 "broadcast",
                 "multibroadcast",
                 "gather"},
                name))
        return input(0);
    if(contains({"add", "sub", "mul", "min", "max"}, name))
    {
        auto x = input(0);
        auto y = input(1);
        if(not x.has_value() or not y.has_value())
            return std::nullopt;
        if(name == "add")
            return *x + *y;
        if(name == "sub")
            return *x - *y;
        if(name == "mul")
            return *x * *y;
        if(name == "min")
            return min(*x, *y);
        return max(*x, *y);
    }
    if(name == "where")
    {
        auto x = input(1);
        auto y = input(2);
        if(not x.has_value() or not y.has_value())
            return std::nullopt;
        return merge(*x, *y);
    }
    if(name == "reduce_sum")
    {
        auto x = input(0);
        if(not x.has_value())
            return std::nullopt;
        const auto& in_shape = ins->inputs().front()->get_shape();
        if(in_shape.dynamic() and not in_shape.symbolic())
            return std::nullopt;
        // A sum of up to n values in [lo, hi], n the most elements reduced into one output
        auto count = [](const shape& sh) {
            auto lens = sh.max_lens();
            return std::accumulate(lens.begin(), lens.end(), std::size_t{1}, std::multiplies<>{});
        };
        auto n = count(in_shape) / std::max<std::size_t>(count(ins->get_shape()), 1);
        return *x * sym::interval{int64_t{0}, static_cast<int64_t>(n)};
    }
    return std::nullopt;
}

struct parse_range : op_parser<parse_range>
{
    std::vector<op_desc> operators() const { return {{"Range"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto start_arg = args[0]->eval();
        auto limit_arg = args[1]->eval();
        auto delta_arg = args[2]->eval();

        if(start_arg.empty() or limit_arg.empty() or delta_arg.empty())
        {
            const auto start = args[0]->sym_eval();
            const auto limit = args[1]->sym_eval();
            const auto delta = args[2]->sym_eval();
            if(not start.empty() and not limit.empty() and not delta.empty())
            {
                const auto start_values = start.get();
                const auto limit_values = limit.get();
                const auto delta_values = delta.get();
                const auto delta_value  = delta_values.size() == 1
                                              ? sym::fixed_value(delta_values.front())
                                              : std::nullopt;
                if(start_values.size() == 1 and limit_values.size() == 1 and
                   delta_value.has_value() and sym::to<int64_t>(*delta_value) == 1)
                {
                    const auto output_length = limit_values.front() - start_values.front();
                    const auto output_bounds = output_length.eval_interval();
                    if(sym::to<int64_t>(output_bounds.min) >= 0)
                    {
                        const shape::dynamic_dimension output_dim{output_length};
                        return info.add_instruction(
                            make_op("dynamic_range", {{"output_dim", to_value(output_dim)}}), args);
                    }
                }
            }
            // A limit only known at run time but bounded: the length becomes a symbol over its
            // range
            auto first_fixed = [](const sym_argument& a) -> std::optional<sym::scalar> {
                if(a.empty() or a.get_shape().elements() != 1)
                    return std::nullopt;
                return sym::fixed_value(a.get()[0]);
            };
            const auto start_value = first_fixed(start);
            const auto delta_value = first_fixed(delta);
            const auto limit_range = value_bounds(args[1]);
            if(limit.empty() and start_value.has_value() and delta_value.has_value() and
               sym::to<int64_t>(*delta_value) == 1 and limit_range.has_value() and
               std::holds_alternative<int64_t>(limit_range->max) and
               std::get<int64_t>(limit_range->max) < std::numeric_limits<int>::max())
            {
                const auto max_length = std::max<int64_t>(
                    sym::to<int64_t>(limit_range->max) - sym::to<int64_t>(*start_value), 0);
                const shape::dynamic_dimension output_dim{
                    sym::var(parser.symbol_names.allocate(info.name), {0, max_length})};
                return info.add_instruction(
                    make_op("dynamic_range",
                            {{"max_output", max_length}, {"output_dim", to_value(output_dim)}}),
                    args);
            }
            return info.add_instruction(make_op("dynamic_range"), args);
        }

        assert(args[0]->get_shape().elements() == 1 and args[1]->get_shape().elements() == 1 and
               args[2]->get_shape().elements() == 1);

        instruction_ref l0;

        visit_all(start_arg, limit_arg, delta_arg)([&](auto start, auto limit, auto delta) {
            auto start_val = start.front();
            auto limit_val = limit.front();
            auto delta_val = delta.front();

            if(not(delta_val > 0 or delta_val < 0))
                MIGRAPHX_THROW("PARSE_RANGE: delta must be non-zero");

            double start_d        = start_val;
            double limit_d        = limit_val;
            double delta_d        = delta_val;
            double num_elements_d = ceil((limit_d - start_d) / delta_d);
            size_t num_elements   = std::max(0.0, num_elements_d);

            using type = decltype(start_val);

            std::vector<type> range_vals(num_elements);

            std::generate(range_vals.begin(), range_vals.end(), [&]() {
                auto result = start_val;
                start_val += delta_val;
                return result;
            });

            l0 = info.add_literal({shape{args[0]->get_shape().type(), {num_elements}}, range_vals});
        });
        return l0;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
