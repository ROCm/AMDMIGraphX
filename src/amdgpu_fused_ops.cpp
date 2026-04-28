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
#include <migraphx/errors.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct amdgpu_gemm_add
{
    std::string kernel_registry = "";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.kernel_registry, "kernel_registry"));
    }

    std::string name() const { return "amdgpu::gemm_add"; }

    value attributes() const { return {{"target", "gpu"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() != 3)
            MIGRAPHX_THROW(name() + ": expected 3 inputs (A, B, C)");

        const auto& a = inputs[0];
        const auto& b = inputs[1];
        const auto& c = inputs[2];

        if(a.ndim() != 2 or b.ndim() != 2 or c.ndim() != 2)
            MIGRAPHX_THROW(name() + ": only rank-2 tensors are currently supported");

        if(a.lens().at(1) != b.lens().at(0))
            MIGRAPHX_THROW(name() + ": incompatible GEMM dimensions");

        shape out{a.type(), {a.lens().at(0), b.lens().at(1)}};
        if(c.lens() != out.lens())
            MIGRAPHX_THROW(name() + ": bias tensor shape must match GEMM output shape");

        return out;
    }
};
MIGRAPHX_REGISTER_OP(amdgpu_gemm_add);

struct amdgpu_gemm_gemm_add
{
    std::string kernel_registry = "";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.kernel_registry, "kernel_registry"));
    }

    std::string name() const { return "amdgpu::gemm_gemm_add"; }

    value attributes() const { return {{"target", "gpu"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() != 4)
            MIGRAPHX_THROW(name() + ": expected 4 inputs (A, B, D, E)");

        const auto& a = inputs[0];
        const auto& b = inputs[1];
        const auto& d = inputs[2];
        const auto& e = inputs[3];

        if(a.ndim() != 2 or b.ndim() != 2 or d.ndim() != 2 or e.ndim() != 2)
            MIGRAPHX_THROW(name() + ": only rank-2 tensors are currently supported");

        if(a.lens().at(1) != b.lens().at(0))
            MIGRAPHX_THROW(name() + ": incompatible first GEMM dimensions");

        const std::size_t m = a.lens().at(0);
        const std::size_t n = b.lens().at(1);
        if(n != d.lens().at(0))
            MIGRAPHX_THROW(name() + ": incompatible second GEMM dimensions");

        shape out{a.type(), {m, d.lens().at(1)}};
        if(e.lens() != out.lens())
            MIGRAPHX_THROW(name() + ": add tensor shape must match second GEMM output shape");

        return out;
    }
};
MIGRAPHX_REGISTER_OP(amdgpu_gemm_gemm_add);

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
