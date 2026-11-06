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

#include <test.hpp>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>

static std::vector<float> gpu_random(const migraphx::shape& s, unsigned long seed)
{
    migraphx::gpu::context ctx{};
    auto dev = migraphx::gpu::gpu_generate_random(ctx, s, seed);
    ctx.finish();
    std::vector<float> v;
    migraphx::gpu::from_gpu(dev).visit([&](auto out) { v.assign(out.begin(), out.end()); });
    return v;
}

static std::vector<uint8_t> gpu_random_bytes(const migraphx::shape& s, unsigned long seed)
{
    migraphx::gpu::context ctx{};
    auto dev = migraphx::gpu::gpu_generate_random(ctx, s, seed);
    ctx.finish();
    auto host     = migraphx::gpu::from_gpu(dev);
    const auto* p = reinterpret_cast<const uint8_t*>(host.data());
    std::vector<uint8_t> bytes(host.get_shape().bytes());
    std::copy(p, p + bytes.size(), bytes.begin());
    return bytes;
}

static bool in_unit_range(const std::vector<float>& v)
{
    return std::all_of(v.begin(), v.end(), [](float x) { return x >= -1.0f and x < 1.0f; });
}

static bool any_nonzero(const std::vector<uint8_t>& v)
{
    return std::any_of(v.begin(), v.end(), [](uint8_t b) { return b != 0; });
}

TEST_CASE(seed_controls_output)
{
    migraphx::shape s{migraphx::shape::float_type, {100003}};
    auto v = gpu_random(s, 1234);
    EXPECT(v.size() == s.elements());
    EXPECT(in_unit_range(v));
    // Same seed reproduces the buffer; well-separated seeds change it
    EXPECT(gpu_random(s, 1234) == v);
    EXPECT(gpu_random(s, 123) != gpu_random(s, 456));
}

TEST_CASE(half_type_is_supported)
{
    migraphx::shape s{migraphx::shape::half_type, {3, 32}};
    auto v = gpu_random(s, 7);
    EXPECT(v.size() == s.elements());
    EXPECT(in_unit_range(v));
}

TEST_CASE(all_computable_types_are_filled)
{
    // Every computable type: byte-level check for fill, seed determinism, and
    // seed dependence (type-agnostic, covers bool/half/bf16/fp8/ints/double).
    const std::vector<migraphx::shape::type_t> types = {migraphx::shape::bool_type,
                                                        migraphx::shape::half_type,
                                                        migraphx::shape::float_type,
                                                        migraphx::shape::double_type,
                                                        migraphx::shape::bf16_type,
                                                        migraphx::shape::int8_type,
                                                        migraphx::shape::uint8_type,
                                                        migraphx::shape::int32_type,
                                                        migraphx::shape::int64_type,
                                                        migraphx::shape::fp8e4m3fn_type,
                                                        migraphx::shape::fp8e5m2_type};
    for(auto t : types)
    {
        migraphx::shape s{t, {4096}};
        auto a = gpu_random_bytes(s, 1234);
        auto b = gpu_random_bytes(s, 1234);
        auto c = gpu_random_bytes(s, 5678);
        EXPECT(a.size() == s.bytes());
        EXPECT(a == b);         // deterministic per seed
        EXPECT(a != c);         // seed controls output
        EXPECT(any_nonzero(a)); // buffer actually filled
    }
}

TEST_CASE(empty_shape_is_noop)
{
    migraphx::gpu::context ctx{};
    // Must not launch / crash for a zero-element shape.
    auto dev = migraphx::gpu::gpu_generate_random(ctx, {migraphx::shape::float_type, {0}}, 0);
    ctx.finish();
    EXPECT(dev.get_shape().elements() == 0);
}

TEST_CASE(tuple_fills_every_sub_buffer)
{
    // A tuple shape is one buffer split into sub-object views (like the time_op
    // path); check every sub-buffer is reached by the recursion.
    migraphx::gpu::context ctx{};
    migraphx::shape tuple_shape{
        std::vector<migraphx::shape>{migraphx::shape{migraphx::shape::float_type, {2, 8}},
                                     migraphx::shape{migraphx::shape::half_type, {3, 5}}}};
    auto dev = migraphx::gpu::gpu_generate_random(ctx, tuple_shape, 99);
    ctx.finish();

    auto subs = migraphx::gpu::from_gpu(dev).get_sub_objects();
    EXPECT(subs.size() == 2);
    for(const auto& sub : subs)
    {
        std::vector<float> v;
        sub.visit([&](auto out) { v.assign(out.begin(), out.end()); });
        EXPECT(v.size() == sub.get_shape().elements());
        EXPECT(in_unit_range(v)); // filled (reached by the recursion), in range
    }
}

TEST_CASE(non_computable_type_is_filled)
{
    // fp4x2 has no visitor; generation must fall back to a raw byte fill instead
    // of throwing "cannot be visited" (the old host path generated uint8 bytes).
    auto bytes = gpu_random_bytes(migraphx::shape{migraphx::shape::fp4x2_type, {64}}, 1);
    EXPECT(any_nonzero(bytes));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
