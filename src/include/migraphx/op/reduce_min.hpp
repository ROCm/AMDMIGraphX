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
#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_MIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_MIN_HPP

#include <migraphx/op/reduce_op.hpp>
#include <migraphx/sym_argument.hpp>

#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_min : reduce_op<reduce_min>
{
    reduce_min() {}
    reduce_min(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x < y ? x : y; };
    }

    auto init() const { return highest(); }

    sym_argument symbolic_compute(const shape& output_shape,
                                  const std::vector<sym_argument>& args) const
    {
        if(args.size() != 1 or args.front().empty() or axes.empty())
            return {};

        const auto input_shape = args.front().get_shape();
        if(input_shape.dynamic() or output_shape.dynamic())
            return {};

        std::vector<std::size_t> batch_lens(output_shape.ndim(), 1);
        this->tune_dims(axes, input_shape.lens(), batch_lens);
        shape batch_shape{input_shape.type(), batch_lens};
        auto input = args.front().get();
        sym_argument result{output_shape};
        auto output = result.get();
        shape_for_each(output_shape, [&](const auto& out_idx) {
            auto data_idx = out_idx;
            std::optional<sym::expr> value;
            shape_for_each(batch_shape, [&](const auto& batch_idx) {
                this->tune_dims(axes, batch_idx, data_idx);
                const auto& current = input(data_idx.begin(), data_idx.end());
                value               = value.has_value() ? sym::min(*value, current) : current;
            });
            if(value.has_value())
                output(out_idx.begin(), out_idx.end()) = *value;
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
