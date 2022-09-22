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
#include <migraphx/gpu/driver/action.hpp>
#include <migraphx/gpu/driver/perf.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

struct compile_op : action<compile_op>
{
    static void apply(const parser& p, const value& v)
    {
        context ctx;
        auto inputs = p.parse_shapes(v.at("inputs"));
        auto op     = gpu::compile_op(v.at("name").to<std::string>(), ctx, inputs, v);
        auto [host_time, device_time] = time_op(ctx, op, inputs, p.get(v, "iterations", 100));
        std::cout << op << ": " << host_time << "ms";
        if(device_time > 0)
            std::cout << ", " << device_time << "ms";
        std::cout << std::endl;
    }
};

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
