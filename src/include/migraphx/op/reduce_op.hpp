/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
        return {{"normalize_axes", normalize}, {"reduce", true}};
    }

    std::vector<int64_t> tune_axes(std::size_t n_dim) const
    {
        auto tuned_axes = axes;
        if(tuned_axes.empty())
        {
            tuned_axes.resize(n_dim);
            std::iota(tuned_axes.begin(), tuned_axes.end(), 0);
        }

        return tuned_axes;
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
        check_shapes{inputs, *this, true}.has(1);
        auto s = inputs.at(0);
        if(s.dynamic())
        {
            auto output_dyn_dims = s.dyn_dims();
            auto tuned_axes      = tune_axes(output_dyn_dims.size());
            for(const auto& axis : tuned_axes)
            {
                output_dyn_dims[axis] = {1, 1};
            }

            return shape{s.type(), output_dyn_dims};
        }
        else
        {
            auto lens       = s.lens();
            auto tuned_axes = tune_axes(lens.size());
            for(const auto& axis : tuned_axes)
            {
                lens[axis] = 1;
            }

            return inputs[0].with_lens(lens);
        }
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
    void reduce(tensor_view<T>& input,
                shape& batch_shape,
                std::vector<int64_t>& tuned_axes,
                std::vector<std::size_t>& out_idx,
                tensor_view<T>& output) const
    {
        using accumulator = accumulator_type<T>;
        auto& self        = static_cast<const Derived&>(*this);
        auto data_idx     = out_idx;
        accumulator val   = self.init();
        shape_for_each(batch_shape, [&](size_t, auto b_idx) {
            this->tune_dims(tuned_axes, b_idx, data_idx);
            accumulator x = input(data_idx.begin(), data_idx.end());
            val           = self.op()(accumulator{self.input()(x)}, val);
        });

        output(out_idx.begin(), out_idx.end()) =
            static_cast<const Derived&>(*this).output(batch_shape)(val);
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        argument result{dyn_out.computed_shape};
        auto arg_lens   = args.front().get_shape().lens();
        auto tuned_axes = tune_axes(arg_lens.size());
        std::vector<std::size_t> batch_lens(dyn_out.computed_shape.lens().size(), 1);
        tune_dims(tuned_axes, arg_lens, batch_lens);
        shape batch_shape{dyn_out.computed_shape.type(), batch_lens};
        visit_all(result, args[0])([&](auto output, auto input) {
            par_for(dyn_out.computed_shape.elements(), [&](auto i) {
                auto out_idx = dyn_out.computed_shape.multi(i);
                this->reduce(input, batch_shape, tuned_axes, out_idx, output);
            });
        });

        return result;
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
