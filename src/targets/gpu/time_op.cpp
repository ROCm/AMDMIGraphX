/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/time.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::vector<argument> generate_arguments(const std::vector<shape>& shapes, unsigned long seed = 0)
{
    std::vector<argument> args;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(args), [&](const auto& s) {
        return to_gpu(generate_argument(s, seed++));
    });
    return args;
}

std::vector<std::vector<argument>> generate_all_arguments(const std::vector<shape>& shapes, int n)
{
    std::vector<std::vector<argument>> result;
    transform(range(n), std::back_inserter(result), [&](auto i) {
        return generate_arguments(shapes, i);
    });
    return result;
}

using milliseconds = std::chrono::duration<double, std::milli>;
template <class F>
double benchmark(context& gctx, const std::vector<shape>& inputs, int n, F run)
{
    auto args  = generate_all_arguments(inputs, std::min(37, n));
    auto start = context::create_event_for_timing();
    auto stop  = context::create_event_for_timing();
    run(args[0]);
    gctx.get_stream().record(start.get());
    for(auto i : range(n))
    {
        run(args[i % args.size()]);
    }
    gctx.get_stream().record(stop.get());
    gctx.finish();
    return context::get_elapsed_ms(start.get(), stop.get()) / n;
}

double benchmark(context& gctx, const std::vector<shape>& inputs, int n, const benchmark_function& run)
{
    return benchmark(gctx, inputs, n, [&](auto&&... xs) {
        run(static_cast<decltype(xs)>(xs)...);
    });
}

double time_op(context& ictx, operation op, const std::vector<shape>& inputs, int n)
{
    migraphx::context ctx = ictx;
    auto& gctx            = any_cast<migraphx::gpu::context>(ctx);
    auto output           = op.compute_shape(inputs);
    op.finalize(ctx, output, inputs);
    return benchmark(gctx, inputs, n, [&](const auto& args) { op.compute(ctx, output, args); });
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
