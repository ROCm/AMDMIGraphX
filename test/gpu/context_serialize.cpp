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
#include <iostream>
#include <vector>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include "test.hpp"

TEST_CASE(gpu_context_serialize)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};

    auto v = ctx.to_value();
    EXPECT(v.size() == 3);

    EXPECT(v.contains("events"));
    EXPECT(v.at("events").without_key().to<std::size_t>() == 0);

    EXPECT(v.contains("streams"));
    EXPECT(v.at("streams").without_key().to<std::size_t>() == 3);

    EXPECT(v.contains("compiled_for"));

    migraphx::gpu::context g_ctx;
    g_ctx.from_value(v);

    auto v1 = g_ctx.to_value();
    EXPECT(v == v1);
}

TEST_CASE(gpu_context_compiled_for)
{
    migraphx::gpu::device_description desc;
    desc.arch   = "gfx1201";
    desc.num_cu = 32;
    migraphx::gpu::context cross_ctx{desc};
    auto v = cross_ctx.to_value();
    auto compiled_for =
        migraphx::from_value<migraphx::gpu::device_description>(v.at("compiled_for"));
    EXPECT(compiled_for.arch == "gfx1201");
    EXPECT(compiled_for.num_cu == 32);
    EXPECT(compiled_for.wavefront_size == 32);

    // A program loaded on another device keeps the device it was compiled for
    migraphx::gpu::context g_ctx;
    g_ctx.from_value(v);
    EXPECT(g_ctx.get_compiled_for() == compiled_for);
    EXPECT(g_ctx.to_value().at("compiled_for") == v.at("compiled_for"));
}

TEST_CASE(gpu_context_cross_compile_schedule_state)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx1100";
    migraphx::gpu::context cross_ctx{desc};
    cross_ctx.restore_schedule_state(5, 2);
    auto v = cross_ctx.to_value();
    EXPECT(v.at("events").without_key().to<std::size_t>() == 5);
    EXPECT(v.at("streams").without_key().to<std::size_t>() == 2);
}

TEST_CASE(context_queue)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};
    // unsafe_get() avoids a type-mismatch throw if MIGRAPHX_ENABLE_NULL_STREAM
    // is set: context::get_queue() returns an untyped any_ptr when the bound
    // stream is nullptr, so get<hipStream_t>() would not yield "false".
    EXPECT(ctx.get_queue().unsafe_get() != nullptr);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
