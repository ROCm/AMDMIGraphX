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
#ifndef MIGRAPHX_GUARD_RTGLIB_CONVOLUTION_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONVOLUTION_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_convolution
{
    op::convolution op;
    shared<convolution_descriptor> cd = nullptr;
    miopenConvFwdAlgorithm_t algo{};
    uint64_t solution_id = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op.padding, "padding"),
                    f(self.op.stride, "stride"),
                    f(self.op.dilation, "dilation"),
                    f(self.op.group, "group"),
                    f(self.op.padding_mode, "padding_mode"),
                    f(self.solution_id, "solution_id"));
    }

    std::string name() const { return "gpu::convolution"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    shape find(context& ctx, const shape& output_shape, std::vector<shape> inputs);
    void finalize(context& ctx, const shape& output_shape, std::vector<shape> inputs);
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
