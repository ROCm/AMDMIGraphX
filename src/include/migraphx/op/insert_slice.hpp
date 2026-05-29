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
#ifndef MIGRAPHX_GUARD_OPERATORS_INSERT_SLICE_HPP
#define MIGRAPHX_GUARD_OPERATORS_INSERT_SLICE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/value.hpp>
#include <migraphx/par_for.hpp>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct insert_slice
{
    std::vector<std::size_t> static_offsets{};
    std::vector<std::size_t> static_sizes{};
    std::vector<std::size_t> static_strides{};
    bool deref_dest = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.static_offsets, "static_offsets"), f(self.static_sizes, "static_sizes"), f(self.static_strides, "static_strides"), f(self.deref_dest, "deref_dest"));
    }

    std::string name() const { return "insert_slice"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has_at_least(2);
        const auto& src  = inputs[0];
        const auto& dest = inputs[1];
        const auto rank  = src.ndim();
        if(inputs.size() > 2)
        {
            const auto& off = inputs[2];
            if(off.ndim() == 1)
            {
                if(off.lens().front() != rank)
                    MIGRAPHX_THROW(
                        "insert_slice: 1D offsets length must equal rank (per-axis offsets)");
            }
            else if(off.ndim() == 2)
            {
                if(off.lens()[1] != rank)
                    MIGRAPHX_THROW(
                        "insert_slice: batched offsets must be shape [batch, rank] (second dim = rank)");
                if(off.lens()[0] != src.lens().front())
                    MIGRAPHX_THROW(
                        "insert_slice: batched offsets batch must match source first dimension");
                if(dest.lens().front() != src.lens().front())
                    MIGRAPHX_THROW(
                        "insert_slice: batched offsets require matching first dimension on src and dest");
            }
            else
            {
                MIGRAPHX_THROW(
                    "insert_slice: offsets tensor must be 1D [rank] or 2D [batch, rank]");
            }
        }
        return inputs[1];
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};

        auto source = args[0];
        auto destination = args[1];
        auto src_shape = source.get_shape();
        auto dest_shape = destination.get_shape();
        auto rank = src_shape.ndim();
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> sizes;
        std::vector<std::size_t> strides;
        const bool tensor_offsets = args.size() > 2;
        auto off_shape            = tensor_offsets ? args[2].get_shape() : shape{};
        const bool batched_offsets =
            tensor_offsets and off_shape.ndim() == 2;
        if(tensor_offsets /* and not batched_offsets */)
        {
            args[2].visit([&](auto offs) {
                offsets.clear();
                std::transform(
                    offs.begin(), offs.end(), std::back_inserter(offsets), [](auto x) {
                        return static_cast<std::size_t>(x);
                    });
            });
        }
        else if(not tensor_offsets)
        {
            offsets = static_offsets;
        }
        if(args.size() > 3)
        {   
            args[3].visit([&](auto szs) {
                std::transform(szs.begin(), szs.end(), std::back_inserter(sizes), [](auto x) { return static_cast<std::size_t>(x); });
            });
        }
        else
        {
            sizes = static_sizes;
        }
        if(args.size() > 4)
        {
            args[4].visit([&](auto strs) {
                std::transform(strs.begin(), strs.end(), std::back_inserter(strides), [](auto x) { return static_cast<std::size_t>(x); });
            });
        }
        else
        {
            strides = static_strides;
        }

        visit_all(result, destination)([&](auto output, auto dest) {
            source.visit([&](auto src) {
                std::copy(dest.begin(), dest.end(), output.begin());
                par_for(src_shape.elements(), [&](auto i) {
                    auto src_idx = src_shape.multi(i);
                    auto dest_idx = dest_shape.multi(i);
                    const std::size_t b = batched_offsets ? src_idx[0] : 0;
                    for(auto j = 0; j < rank; j++) {
                        dest_idx[j] = (src_idx[j] * strides[j]) + offsets[(b * rank) + j];
                    }
                    if(deref_dest)
                    {
                        using src_type = typename std::remove_cv_t<typename decltype(src)::value_type>;
                        auto addr = static_cast<uintptr_t>(output[dest_idx]);
                        auto* ptr = reinterpret_cast<src_type*>(addr);
                        *ptr      = src[src_idx];
                    }
                    else
                    {
                        output[dest_idx] = src[src_idx];
                    }
                });
            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
