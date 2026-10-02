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
#include <migraphx/gpu/gptoss_moe.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/device/moe.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/half.hpp>
#include <fstream>
#include <cstdlib>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gptoss_moe::compute_shape(std::vector<shape> inputs) const
{
    // After lowering, an output buffer is appended (9 inputs); before that there
    // are the 8 op inputs. Either way report the op's output shape (= hidden shape).
    if(inputs.size() < 8)
        MIGRAPHX_THROW("gpu::gptoss_moe: expected >=8 inputs, got " +
                       std::to_string(inputs.size()));
    return inputs.front();
}

argument
hip_gptoss_moe::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    // First-light: throw (propagates across the DLL boundary to the runner's
    // catch + stderr; fprintf from migraphx_device.dll does not).
    if(args.size() != 9)
        MIGRAPHX_THROW("gpu::gptoss_moe::compute: expected 9 args (8 inputs + output), got " +
                       std::to_string(args.size()));

    const auto& hidden  = args[0];
    const auto& router  = args[1];
    const auto& fc1_w   = args[2];
    const auto& fc1_s   = args[3];
    const auto& fc2_w   = args[4];
    const auto& fc2_s   = args[5];
    const auto& fc1_b   = args[6];
    const auto& fc2_b   = args[7];
    const auto& output  = args[8];

    // Surface the shapes/types we actually received so a mismatch is diagnosable.
    auto sstr = [](const shape& s) { return s.type_string() + to_string_range(s.lens()); };
    if(hidden.get_shape().type() != shape::float_type)
        MIGRAPHX_THROW("gpu::gptoss_moe: hidden must be float (got " + sstr(hidden.get_shape()) +
                       ")");
    if(fc1_w.get_shape().type() != shape::uint32_type or
       fc2_w.get_shape().type() != shape::uint32_type)
        MIGRAPHX_THROW("gpu::gptoss_moe: fc weights must be uint32 (got fc1=" +
                       sstr(fc1_w.get_shape()) + " fc2=" + sstr(fc2_w.get_shape()) + ")");

    const int S      = static_cast<int>(hidden.get_shape().lens()[0]);
    const int E      = op.num_experts;
    const int top_k  = op.top_k;
    // Worst case: every token routes to top_k distinct experts.
    const int maxtok = S * top_k;


    // Scratch (device). B1: persistent grow-only caches instead of a fresh
    // allocate_gpu (= hipMalloc) on every call. The MoE op runs 24x/token; with
    // per-call allocation that was ~5 synchronous hipMalloc/free * 24 layers =
    // ~120 device allocs/token, each a device-wide sync point — the dominant
    // host-dispatch cost (Phase A roofline §11). Each scratch buffer keeps its own
    // thread_local pointer, grow-only, reused across layers and decode steps;
    // steady decode (fixed S) allocates nothing after warmup. Wrapped as NON-OWNING
    // arguments (argument(shape,T*)) so they are not freed on scope exit. Buffers
    // needing init (etok_cnt, output) are explicitly zeroed inside
    // device::gptoss_moe, so reuse across calls is correctness-neutral. Process-
    // lifetime cache (not freed at teardown — avoids racing HIP runtime shutdown).
    struct moe_scratch_slot
    {
        void* ptr      = nullptr;
        std::size_t sz = 0;
        void* get(std::size_t bytes)
        {
            if(sz < bytes)
            {
                if(ptr != nullptr)
                    (void)hipFree(ptr);
                if(hipMalloc(&ptr, bytes) != hipSuccess)
                    MIGRAPHX_THROW("gpu::gptoss_moe: scratch hipMalloc failed");
                sz = bytes;
            }
            return ptr;
        }
    };
    static thread_local moe_scratch_slot s_topk_w, s_topk_e, s_etok_ids, s_etok_cnt;

    shape sh_topk_w{shape::float_type, {static_cast<std::size_t>(S), static_cast<std::size_t>(top_k)}};
    shape sh_topk_e{shape::int32_type, {static_cast<std::size_t>(S), static_cast<std::size_t>(top_k)}};
    shape sh_etok_ids{shape::int32_type, {static_cast<std::size_t>(E), static_cast<std::size_t>(maxtok)}};
    shape sh_etok_cnt{shape::int32_type, {static_cast<std::size_t>(E)}};
    argument topk_w{sh_topk_w, reinterpret_cast<float*>(s_topk_w.get(sh_topk_w.bytes()))};
    argument topk_e{sh_topk_e, reinterpret_cast<std::int32_t*>(s_topk_e.get(sh_topk_e.bytes()))};
    argument etok_ids{sh_etok_ids, reinterpret_cast<std::int32_t*>(s_etok_ids.get(sh_etok_ids.bytes()))};
    argument etok_cnt{sh_etok_cnt, reinterpret_cast<std::int32_t*>(s_etok_cnt.get(sh_etok_cnt.bytes()))};

    device::moe_params p;
    p.num_tokens            = S;
    p.hidden_size           = op.hidden_size;
    p.intermediate_size     = op.intermediate_size;
    p.num_experts           = E;
    p.top_k                 = top_k;
    p.max_tokens_per_expert = maxtok;
    p.swiglu_alpha          = op.swiglu_alpha;
    p.swiglu_beta           = op.swiglu_beta;
    p.swiglu_limit          = op.swiglu_limit;

    device::gptoss_moe(ctx.get_stream().get(),
                       output,
                       hidden,
                       router,
                       fc1_w,
                       fc1_s,
                       fc2_w,
                       fc2_s,
                       fc1_b,
                       fc2_b,
                       topk_w,
                       topk_e,
                       etok_ids,
                       etok_cnt,
                       p);

    return output;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
