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
#include <migraphx/gpu/hip_graph.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/program.hpp>
#include <migraphx/context.hpp>
#include <migraphx/argument.hpp>
#include <iostream>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void enable_hip_graph(program& p)
{
    auto& base_ctx = p.get_context();
    auto* gpu_ctx  = any_cast<context>(&base_ctx);
    if(gpu_ctx == nullptr)
    {
        std::cerr << "[HIP Graph] Warning: program context is not a GPU context, "
                     "graph eval not enabled."
                  << std::endl;
        return;
    }

    if(not gpu_ctx->hip_graph_enabled())
    {
        return;
    }

    std::cerr << "[HIP Graph] Enabling HIP Graph capture/replay for program." << std::endl;

    // Three phases: warmup -> capture -> replay
    struct graph_state
    {
        context* ctx                          = nullptr;
        std::vector<argument> cached_outputs  = {};
        bool warmup_done                      = false;
        bool captured                         = false;
        bool capture_failed                   = false;
        std::size_t launch_count              = 0;
    };

    auto state = std::make_shared<graph_state>();
    state->ctx = gpu_ctx;

    p.set_graph_eval(
        [state](const std::function<std::vector<argument>()>& normal_eval)
            -> std::vector<argument> {
            auto* ctx = state->ctx;

            if(state->capture_failed)
            {
                return normal_eval();
            }

            // Replay captured graph
            if(state->captured && ctx->has_graph())
            {
                ctx->launch_graph();
                state->launch_count++;
                if(state->launch_count == 1 || (state->launch_count % 1000) == 0)
                {
                    std::cerr << "[HIP Graph] Replay #" << state->launch_count
                              << " (hipGraphLaunch)" << std::endl;
                }
                return state->cached_outputs;
            }

            // Warmup: force lazy initialization
            if(not state->warmup_done)
            {
                std::cerr << "[HIP Graph] Warmup eval (initializing GPU state)..."
                          << std::endl;
                auto result        = normal_eval();
                state->warmup_done = true;
                return result;
            }

            // Capture
            std::cerr << "[HIP Graph] Capturing graph..." << std::endl;

            try
            {
                ctx->begin_graph_capture();
                auto result = normal_eval();

                ctx->end_graph_capture();

                state->cached_outputs = result;
                state->captured       = true;

                std::cerr << "[HIP Graph] Graph captured successfully." << std::endl;

                ctx->launch_graph();

                return result;
            }
            catch(const std::exception& e)
            {
                std::cerr << "[HIP Graph] Capture failed: " << e.what()
                          << "\nPermanently disabling HIP Graph for this program."
                          << std::endl;

                if(ctx->is_capturing())
                {
                    hipGraph_t discard = nullptr;
                    hipStreamEndCapture(ctx->get_stream().get(), &discard);
                    if(discard != nullptr)
                        hipGraphDestroy(discard);
                }

                ctx->reset_graph();
                state->capture_failed = true;
                return normal_eval();
            }
        });
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
