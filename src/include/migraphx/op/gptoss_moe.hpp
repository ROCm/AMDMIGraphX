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
#ifndef MIGRAPHX_GUARD_OPERATORS_GPTOSS_MOE_HPP
#define MIGRAPHX_GUARD_OPERATORS_GPTOSS_MOE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// GPT-OSS sparse top-4 MoE layer (single fused op).
//
// Inputs (kernel-native layout; weights repacked by the model rewriter):
//   0 hidden_states [S, hidden]            (io_dtype; converted to f32 on GPU)
//   1 router_logits [S, num_experts]       (f32)
//   2 fc1_weights   [E, 2*inter, hidden/8] (uint32, MatMulNBits INT4)
//   3 fc1_scales    [E, 2*inter, hidden/32](f32)
//   4 fc2_weights   [E, hidden, inter/8]   (uint32)
//   5 fc2_scales    [E, hidden, inter/32]  (f32)
// Output: [S, hidden] (io_dtype, same as input 0)
struct gptoss_moe
{
    int num_experts       = 32;
    int top_k             = 4;
    int hidden_size       = 2880;
    int intermediate_size = 2880;
    float swiglu_alpha    = 1.702f;
    float swiglu_beta     = 1.0f;
    float swiglu_limit    = 7.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.num_experts, "num_experts"),
                    f(self.top_k, "top_k"),
                    f(self.hidden_size, "hidden_size"),
                    f(self.intermediate_size, "intermediate_size"),
                    f(self.swiglu_alpha, "swiglu_alpha"),
                    f(self.swiglu_beta, "swiglu_beta"),
                    f(self.swiglu_limit, "swiglu_limit"));
    }

    std::string name() const { return "gptoss_moe"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(8);
        // Output matches hidden_states (input 0): [S, hidden], same dtype.
        return inputs.front();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
