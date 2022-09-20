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
#include <migraphx/gpu/driver/perf.hpp>
#include <migraphx/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/time.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

std::vector<argument> generate_arguments(const std::vector<shape>& shapes, unsigned long seed = 0)
{
    std::vector<argument> args;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(args), [&](auto& s) {
        return to_gpu(generate_argument(s, seed++));
    });
    return args;
}

using milliseconds = std::chrono::duration<double, std::milli>;
std::pair<double, double>
time_op(context& ictx, operation op, const std::vector<shape>& inputs, int n)
{

    // TODO: Use std::ref
    migraphx::context ctx = ictx;
    auto& gctx            = any_cast<migraphx::gpu::context>(ctx);
    auto output           = op.compute_shape(inputs);
    op.finalize(ctx, output, inputs);
    auto args = generate_arguments(inputs);
    auto run  = [&] {
        op.compute(ctx, output, args);
        ctx.finish();
    };
    gctx.enable_perf_measurement();
    run();
    double host_time   = 0.0;
    double device_time = 0.0;
    for(auto i : range(n))
    {
        (void)i;
        host_time += time<milliseconds>(run);
        device_time += gctx.get_elapsed_ms();
    }
    return std::make_pair(host_time / n, device_time / n);
}

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
