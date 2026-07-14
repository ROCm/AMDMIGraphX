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
#ifndef MIGRAPHX_GUARD_GPU_REWRITE_SSD_HPP
#define MIGRAPHX_GUARD_GPU_REWRITE_SSD_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

/**
 * Rewrite the SSD post-processing tail so it runs with static shapes until the outputs.
 *
 * The ONNX NMS parser emits a variable-end slice(get_tuple_elem[0](nms), get_tuple_elem[1](nms))
 * that trims the zero-padded selected indices to num_selected, which makes the whole downstream
 * gather/topk chain dynamic. This pass:
 *   - bypasses that dynamic slice and masks the gathered scores so the padded rows can never win
 *     the topk (sentinel = lowest/highest representable value for a largest/smallest topk),
 *   - rewrites the topk to sort only the maximum k the model keeps (the literal in the k-calc
 *     concat before reduce_min), and
 *   - moves the num_selected-based trim to the module outputs (min(cap, num_selected)).
 */
struct MIGRAPHX_GPU_EXPORT rewrite_ssd
{
    std::string name() const { return "gpu::rewrite_ssd"; }
    void apply(module& m) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_REWRITE_SSD_HPP
