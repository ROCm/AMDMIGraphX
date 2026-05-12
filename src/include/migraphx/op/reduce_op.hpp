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
#ifndef MIGRAPHX_GUARD_OPERATORS_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_OP_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct lowest
{
    template <class T>
    operator T() const
    {
        return std::numeric_limits<T>::lowest();
    }
};

struct highest
{
    template <class T>
    operator T() const
    {
        return std::numeric_limits<T>::max();
    }
};

struct zero
{
    template <class T>
    operator T() const
    {
        return T{0};
    }
};

struct one
{
    template <class T>
    operator T() const
    {
        return T{1};
    }
};

template <class Derived>
struct reduce_op : op_name<Derived>
{
    std::vector<std::int64_t> axes{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize},
                {"reduce", true},
                {"fillcolor", "#8470FF" /* lightslateblue */}};
    }

    shape collapse_reduced_axes(const shape& original_shape,
                                const std::vector<int64_t>& reduce_axes) const
    {
        auto lens = original_shape.lens();
        for(const auto a : reduce_axes)
        {
            lens[a] = 1;
        }

        return original_shape.with_lens(lens);
    }

    // Variable-axes form (axes attribute empty): every axis is potentially reduced, so each
    // output dim is {1, upper_bound}. Works uniformly for static, range-dynamic, and symbolic
    // inputs -- get_interval().max evaluates the sym::expr upper bound when the dim is symbolic.
    shape variable_axes_compute_shape(const shape& s0) const
    {
        auto dims = s0.to_dynamic().dyn_dims();
        std::transform(dims.begin(), dims.end(), dims.begin(), [](const auto& d) {
            return shape::dynamic_dimension{1, d.get_interval().max};
        });
        return {s0.type(), dims};
    }

    // Fixed-axes form for range-based dynamic input: set the reduced axes to {1,1}.
    shape range_compute_shape(const shape& s0) const
    {
        auto dims = s0.dyn_dims();
        for(auto a : axes)
        {
            dims[a] = {1, 1};
        }
        return {s0.type(), dims};
    }

    // Fixed-axes form for static or symbolic input: build the output symbolically with the
    // reduced axes set to lit(1), then collapse to a static shape if the input was static.
    shape symbolic_compute_shape(const shape& s0) const
    {
        auto sym_in = s0.to_symbolic();
        auto dds    = sym_in.dyn_dims();
        for(auto a : axes)
        {
            dds[a] = shape::dynamic_dimension{sym::lit(1)};
        }
        shape result{s0.type(), dds};
        if(not s0.symbolic())
            return result.to_static();
        return result;
    }

    /**
     * @brief returns a shape in which the axis or axes named
     * for reduction by this op are set, to size 1.
     *
     * @param inputs list of input shapes
     * @return shape
     */
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        auto expected_arg_count = axes.empty() ? 2 : 1;
        check_shapes{inputs, *this, true}.has(expected_arg_count);
        const auto& s0 = inputs[0];

        if(axes.empty())
            return variable_axes_compute_shape(s0);
        if(s0.dynamic() and not s0.symbolic())
            return range_compute_shape(s0);
        return symbolic_compute_shape(s0);
    }

    template <class T>
    void tune_dims(const std::vector<int64_t>& tuned_axes,
                   const std::vector<T>& in_lens,
                   std::vector<T>& out_lens) const
    {
        for(const auto& axis : tuned_axes)
        {
            out_lens[axis] = in_lens[axis];
        }
    }

    template <class T>
    void reduce(const tensor_view<T>& input,
                const shape& batch_shape,
                const std::vector<int64_t>& tuned_axes,
                const std::vector<std::size_t>& out_idx,
                tensor_view<T>& output) const
    {
        using accumulator = accumulator_type<T>;
        auto& self        = static_cast<const Derived&>(*this);
        auto data_idx     = out_idx;
        accumulator val   = self.init();
        shape_for_each(batch_shape, [&](const auto& b_idx) {
            this->tune_dims(tuned_axes, b_idx, data_idx);
            accumulator x = input(data_idx.begin(), data_idx.end());
            val           = self.op()(accumulator{self.input()(x)}, val);
        });

        output(out_idx.begin(), out_idx.end()) =
            static_cast<const Derived&>(*this).output(batch_shape)(val);
    }

    argument reduce(const shape& computed_shape,
                    const std::vector<int64_t>& reduce_axes,
                    argument& data_arg) const
    {
        std::vector<std::size_t> batch_lens(computed_shape.ndim(), 1);
        auto arg_lens = data_arg.get_shape().lens();
        tune_dims(reduce_axes, arg_lens, batch_lens);
        shape batch_shape{computed_shape.type(), batch_lens};
        argument result{computed_shape};

        visit_all(result, data_arg)([&](auto output, auto input) {
            par_for(computed_shape.elements(), [&](auto i) {
                auto out_idx = computed_shape.multi(i);
                this->reduce(input, batch_shape, reduce_axes, out_idx, output);
            });
        });

        return result;
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        auto&& data_arg = args[0];
        // cppcheck-suppress knownConditionTrueFalse
        if(not axes.empty())
            return reduce(dyn_out.computed_shape, axes, data_arg);

        if(args[1].get_shape().elements() == 0)
            return args[0];

        std::vector<int64_t> reduce_axes;
        args[1].visit([&](auto&& s) { reduce_axes.assign(s.begin(), s.end()); });
        const auto result_shape = collapse_reduced_axes(data_arg.get_shape(), reduce_axes);

        return reduce(result_shape, reduce_axes, data_arg);
    }

    auto init() const { return zero(); }

    auto input() const
    {
        return [](auto val) { return val; };
    }

    auto output(const shape&) const
    {
        return [](auto val) { return val; };
    }

    reduce_op() {}
    reduce_op(std::vector<int64_t> ax) : axes(std::move(ax)) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
