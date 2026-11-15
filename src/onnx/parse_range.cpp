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
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

static const sym::interval& bool_bounds()
{
    static const sym::interval result{int64_t{0}, int64_t{1}};
    return result;
}

static sym::interval merge_bounds(const sym::interval& a, const sym::interval& b)
{
    return sym::interval{sym::scalar_min(a.min, b.min), sym::scalar_max(a.max, b.max)};
}

// Whether every value in `bounds` is held by type `t` without wrapping or rounding: within its
// range for an integer type, and within the exact integers for float and double
static bool fits_type(shape::type_t t, const sym::interval& bounds)
{
    const double lo = sym::to<double>(bounds.min);
    const double hi = sym::to<double>(bounds.max);
    if(not std::isfinite(lo) or not std::isfinite(hi))
        return false;
    double lowest  = 0;
    double highest = 0;
    if(t == shape::float_type or t == shape::double_type)
    {
        highest = std::ldexp(1.0, t == shape::float_type ? 24 : 53);
        lowest  = -highest;
    }
    else if(shape::is_integral(t))
    {
        shape{t}.visit_type([&](auto as) {
            lowest  = static_cast<double>(as.min());
            highest = static_cast<double>(as.max());
        });
    }
    else
    {
        return false;
    }
    return lowest <= lo and hi <= highest;
}

// The range of a convert to type `t` of values in the range `x`, of type `from`
static std::optional<sym::interval>
convert_bounds(shape::type_t from, shape::type_t t, const std::optional<sym::interval>& x)
{
    if(t == shape::bool_type)
        return bool_bounds();
    if(not x.has_value() or not shape::is_integral(t) or shape::is_integral(from))
        return x;
    // A float truncates toward zero, so it stays within the floor and ceil of its range
    const sym::interval truncated{std::floor(sym::to<double>(x->min)),
                                  std::ceil(sym::to<double>(x->max))};
    if(not fits_type(t, truncated))
        return std::nullopt;
    return sym::interval{static_cast<int64_t>(sym::to<double>(truncated.min)),
                         static_cast<int64_t>(sym::to<double>(truncated.max))};
}

// The range of a binary op, or of a where choosing between `x` and `y`
static std::optional<sym::interval> binary_bounds(const std::string& name,
                                                  const std::optional<sym::interval>& x,
                                                  const std::optional<sym::interval>& y)
{
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
    if(name == "max")
        return max(*x, *y);
    return merge_bounds(*x, *y);
}

// The range of a reduce_sum of values in the range `x`
static std::optional<sym::interval> reduce_sum_bounds(instruction_ref ins,
                                                      const std::optional<sym::interval>& x)
{
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

// The range of values an instruction can hold, inferred from the ranges of its inputs, or nullopt
// when it cannot be bounded. It bounds a value only known at run time, such as a count of selected
// elements, so the length that depends on it can be a bounded symbol. Each instruction is
// evaluated once.
struct value_bounds
{
    std::unordered_map<instruction_ref, std::optional<sym::interval>> cache = {};

    std::optional<sym::interval> operator()(instruction_ref ins, int depth = 0)
    {
        auto it = cache.find(ins);
        if(it != cache.end())
            return it->second;
        auto result = compute(ins, depth);
        // A value its type cannot hold would wrap or round at run time
        if(result.has_value() and not fits_type(ins->get_shape().type(), *result))
            result = std::nullopt;
        cache.emplace(ins, result);
        return result;
    }

    private:
    std::optional<sym::interval> compute(instruction_ref ins, int depth)
    {
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
                result = result.has_value() ? merge_bounds(*result, i) : i;
            }
            return result;
        }
        if(depth > 64 or ins->inputs().empty())
            return std::nullopt;
        const auto& name = ins->name();
        auto input       = [&](std::size_t i) { return (*this)(ins->inputs().at(i), depth + 1); };
        if(contains({"equal", "less", "greater", "not", "logical_and", "logical_or", "logical_xor"},
                    name))
            return bool_bounds();
        if(name == "convert")
            return convert_bounds(ins->inputs().front()->get_shape().type(),
                                  s.type(),
                                  s.type() == shape::bool_type ? std::nullopt : input(0));
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
            return binary_bounds(name, input(0), input(1));
        if(name == "where")
            return binary_bounds(name, input(1), input(2));
        if(name == "reduce_sum")
            return reduce_sum_bounds(ins, input(0));
        return std::nullopt;
    }
};

// The length of a range from a whole-number `start` up to a limit in `limit_range`, with a delta
// of 1, or nullopt when it is not bounded by an int
static std::optional<int64_t> bounded_length(const sym::scalar& start,
                                             const std::optional<sym::interval>& limit_range)
{
    const double start_d = sym::to<double>(start);
    const double whole   = std::trunc(start_d);
    if(not limit_range.has_value() or not std::isfinite(start_d) or whole < start_d or
       start_d < whole)
        return std::nullopt;
    const double length = std::ceil(sym::to<double>(limit_range->max) - start_d);
    if(not(length < std::numeric_limits<int>::max()))
        return std::nullopt;
    return std::max<int64_t>(static_cast<int64_t>(length), 0);
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
            const auto max_length  = start_value.has_value()
                                         ? bounded_length(*start_value, value_bounds{}(args[1]))
                                         : std::nullopt;
            if(limit.empty() and max_length.has_value() and delta_value.has_value() and
               sym::to<int64_t>(*delta_value) == 1)
            {
                const shape::dynamic_dimension output_dim{
                    sym::var(parser.symbol_names.allocate(info.name), {0, *max_length})};
                return info.add_instruction(
                    make_op("dynamic_range",
                            {{"max_output", *max_length}, {"output_dim", to_value(output_dim)}}),
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
