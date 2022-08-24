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
#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTERND_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTERND_OP_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct scatternd_op : op_name<Derived>
{
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        auto r         = inputs.front().lens().size();
        auto q         = inputs.at(1).lens().size();
        auto k         = inputs.at(1).lens().back();
        auto ind_lens  = inputs.at(1).lens();
        auto upd_lens  = inputs.back().lens();
        auto data_lens = inputs.front().lens();
        if(k > r)
            MIGRAPHX_THROW("ScatterND: index of size " + std::to_string(k) +
                           " is too large for tensor of rank " + std::to_string(r));
        if(not(std::equal(ind_lens.begin(), ind_lens.begin() + q - 1, upd_lens.begin()) and
               std::equal(data_lens.begin() + k, data_lens.end(), upd_lens.begin() + q - 1)))
            MIGRAPHX_THROW("ScatterND: incorrect update shape. update.lens != indices.lens[0:q-1] "
                           "++ data.lens[k:r-1]");
        auto s = inputs.front();
        if(s.broadcasted())
        {
            return {s.type(), s.lens()};
        }
        else
        {
            return s.with_lens(s.lens());
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto& self = static_cast<const Derived&>(*this);
        visit_all(result, args[0], args[2])([&](auto output, auto data, auto updates) {
            std::copy(data.begin(), data.end(), output.begin());
            args[1].visit([&](auto indices) {
                auto updates_shape = updates.get_shape();
                auto updates_std   = shape{updates_shape.type(), updates_shape.lens()};
                auto indices_shape = indices.get_shape();
                auto k             = indices_shape.lens().back();
                auto q             = indices_shape.lens().size();
                auto r             = output_shape.lens().size();
                par_for(updates_shape.elements(), [&](const auto i) {
                    auto updates_idx = updates_std.multi(i);
                    std::vector<std::size_t> indices_idx(q, 0);
                    std::copy(
                        updates_idx.begin(), updates_idx.begin() + q - 1, indices_idx.begin());
                    auto index_start = indices.begin() +
                                       indices_shape.index(indices_idx.begin(), indices_idx.end());
                    auto index_end = index_start + k;

                    std::vector<std::size_t> out_idx(r, 0);
                    std::copy(index_start, index_end, out_idx.begin());
                    std::copy(updates_idx.begin() + q - 1, updates_idx.end(), out_idx.begin() + k);

                    self.reduction()(output[output_shape.index(out_idx)], updates[i]);
                });
            });
        });

        return result;
    }

    auto init() const {}
    scatternd_op() {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
