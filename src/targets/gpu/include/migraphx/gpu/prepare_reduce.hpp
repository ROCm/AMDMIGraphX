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
 *
 */
#ifndef MIGRAPHX_GUARD_GPU_PREPARE_REDUCE_HPP
#define MIGRAPHX_GUARD_GPU_PREPARE_REDUCE_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/operation.hpp>
#include <migraphx/shape.hpp>
#include <string>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

struct parallel_reduce
{
    operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::parallel_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        std::vector<shape> result;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(result), [&](auto input) {
            return op.compute_shape({input});
        });
        return shape{result};
    }
};

struct arg_reduce
{
    operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::arg_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        auto index_shape = op.compute_shape({inputs.front()});
        auto value_shape = index_shape.with_type(inputs.front().type());
        return shape{{value_shape, index_shape}};
    }
};

struct make_indices
{
    std::size_t size = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    std::string name() const { return "gpu::make_indices"; }

    shape compute_shape(const std::vector<shape>&) const
    {
        return shape{shape::uint32_type, {size}};
    }
};

struct MIGRAPHX_GPU_EXPORT prepare_reduce
{
    std::string name() const { return "gpu::prepare_reduce"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_PREPARE_REDUCE_HPP
