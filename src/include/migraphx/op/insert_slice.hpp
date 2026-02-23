/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_INSERT_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_INSERT_SLICE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Insert a tensor (src) into another tensor (dest) at given offset(s) along one axis.
 *
 * Inputs: src, offsets, dest (in that order).
 * - src: tensor to copy from.
 * - offsets: scalar or tensor of offsets along axis (one per "outer" slice).
 * - dest: tensor to copy into; output shape = dest shape (in-place).
 *
 * Attributes:
 * - axis: dimension along which the offset is applied.
 * - deref: when true, dest must be unsigned integer type; the value of dest
 *   is dereferenced and the write is performed there.
 *
 * Constraint: dest.lens()[axis] >= src.lens()[axis].
 */
struct insert_slice
{
    std::size_t axis = 0;
    bool deref       = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.deref, "deref"));
    }

    std::string name() const { return "insert_slice"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);

        const shape& src_shape     = inputs[0];
        const shape& offsets_shape = inputs[1];
        const shape& dest_shape    = inputs[2];

        if(axis >= dest_shape.ndim())
            MIGRAPHX_THROW(name() + ": axis " + std::to_string(axis) +
                           " must be less than dest ndim " + std::to_string(dest_shape.ndim()));

        if(src_shape.ndim() != dest_shape.ndim())
            MIGRAPHX_THROW(name() + ": src and dest must have same number of dimensions");

        const std::size_t dest_axis_len = dest_shape.lens()[axis];
        const std::size_t src_axis_len  = src_shape.lens()[axis];
        if(dest_axis_len < src_axis_len)
            MIGRAPHX_THROW(name() + ": dest dimension at axis must be >= src dimension at axis (" +
                           std::to_string(dest_axis_len) + " < " + std::to_string(src_axis_len) +
                           ")");

        for(std::size_t i = 0; i < dest_shape.ndim(); ++i)
        {
            if(i == axis)
                continue;
            if(src_shape.lens()[i] != dest_shape.lens()[i])
                MIGRAPHX_THROW(name() + ": src and dest must match on all dimensions except axis");
        }

        if(deref)
        {
            const auto t = dest_shape.type();
            if(t != shape::uint8_type and t != shape::uint16_type and t != shape::uint32_type and
               t != shape::uint64_type)
                MIGRAPHX_THROW(name() +
                              ": when deref is true, dest must have unsigned integer type");
        }

        std::size_t num_outer = 1;
        for(std::size_t i = 0; i < axis; ++i)
            num_outer *= dest_shape.lens()[i];
        const std::size_t num_offsets = offsets_shape.elements();
        if(num_offsets != 1 and num_offsets != num_outer)
            MIGRAPHX_THROW(name() + ": offsets must have 1 element or " +
                           std::to_string(num_outer) + " elements (product of dest dims before "
                           "axis), got " +
                           std::to_string(num_offsets));

        return dest_shape;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument dest = args[2];
        if(deref)
        {
            MIGRAPHX_THROW(name() + ": deref=true is not implemented for CPU execution");
        }

        const shape& src_shape     = args[0].get_shape();
        const shape& offsets_shape = args[1].get_shape();
        const shape& dest_shape    = args[2].get_shape();

        std::size_t num_outer = 1;
        for(std::size_t i = 0; i < axis; ++i)
            num_outer *= dest_shape.lens()[i];
        const bool single_offset = (offsets_shape.elements() == 1);

        visit_all(args[0], dest)([&](auto src_view, auto dest_view) {
            args[1].visit([&](auto offsets_view) {
                shape_for_each(src_shape, [&](const std::vector<std::size_t>& src_idx, std::size_t) {
                    std::size_t outer_linear = 0;
                    std::size_t stride       = 1;
                    for(std::size_t i = axis; i-- > 0;)
                    {
                        outer_linear += src_idx[i] * stride;
                        stride *= dest_shape.lens()[i];
                    }
                    std::size_t off =
                        single_offset ? static_cast<std::size_t>(offsets_view[0])
                                      : static_cast<std::size_t>(offsets_view[outer_linear]);

                    std::vector<std::size_t> dest_idx = src_idx;
                    dest_idx[axis]                    = off + src_idx[axis];

                    dest_view[dest_shape.index(dest_idx.begin(), dest_idx.end())] =
                        src_view[src_shape.index(src_idx.begin(), src_idx.end())];
                });
            });
        });

        return dest;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
