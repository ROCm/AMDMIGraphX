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

#include <test.hpp>
#include <vector>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/target.hpp>

static std::vector<float> read_gpu(const migraphx::argument& dev)
{
    std::vector<float> v;
    migraphx::gpu::from_gpu(dev).visit([&](auto out) { v.assign(out.begin(), out.end()); });
    return v;
}

TEST_CASE(tuple_from_gpu)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {2, 4}};
    std::vector<float> p1_data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    std::vector<int> p2_data   = {1, 2, 3, 4, 5, 6, 7, 8};
    auto p1                    = migraphx::argument{s1, p1_data.data()};
    auto p2                    = migraphx::argument{s2, p2_data.data()};
    auto p1_gpu                = migraphx::gpu::to_gpu(p1);
    auto p2_gpu                = migraphx::gpu::to_gpu(p2);
    auto p_tuple               = migraphx::gpu::from_gpu(migraphx::argument({p1_gpu, p2_gpu}));
    std::vector<migraphx::argument> results = p_tuple.get_sub_objects();
    std::vector<float> result1;
    results[0].visit([&](auto output) { result1.assign(output.begin(), output.end()); });
    std::vector<int> result2;
    results[1].visit([&](auto output) { result2.assign(output.begin(), output.end()); });
    EXPECT(result1 == p1_data);
    EXPECT(result2 == p2_data);
}

TEST_CASE(tuple_to_gpu)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {2, 4}};
    std::vector<float> p1_data              = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    std::vector<int> p2_data                = {1, 2, 3, 4, 5, 6, 7, 8};
    auto p1                                 = migraphx::argument{s1, p1_data.data()};
    auto p2                                 = migraphx::argument{s2, p2_data.data()};
    auto p_gpu                              = migraphx::gpu::to_gpu(migraphx::argument({p1, p2}));
    auto p_host                             = migraphx::gpu::from_gpu(p_gpu);
    std::vector<migraphx::argument> results = p_host.get_sub_objects();
    std::vector<float> result1;
    results[0].visit([&](auto output) { result1.assign(output.begin(), output.end()); });
    std::vector<int> result2;
    results[1].visit([&](auto output) { result2.assign(output.begin(), output.end()); });
    EXPECT(result1 == p1_data);
    EXPECT(result2 == p2_data);
}

TEST_CASE(fill_packed)
{
    migraphx::gpu::context ctx{};
    migraphx::shape s{migraphx::shape::float_type, {2, 6}};
    std::vector<float> ones(s.elements(), 1.0f);
    auto buffer = migraphx::gpu::to_gpu(migraphx::argument{s, ones.data()});

    migraphx::gpu::gpu_fill(ctx, buffer, 0);
    ctx.finish();

    auto result = read_gpu(buffer);
    EXPECT(result == std::vector<float>(s.elements(), 0.0f));
}

TEST_CASE(fill_nonpacked)
{
    // Fill a non-packed 2x3 view (left columns) of a 2x6 buffer; the packed
    // memset path cannot be used, so this covers the device::fill path
    migraphx::gpu::context ctx{};
    migraphx::shape buffer_shape{migraphx::shape::float_type, {2, 6}};
    std::vector<float> ones(buffer_shape.elements(), 1.0f);
    auto buffer = migraphx::gpu::to_gpu(migraphx::argument{buffer_shape, ones.data()});

    migraphx::shape view_shape{migraphx::shape::float_type, {2, 3}, {6, 1}};
    EXPECT(not view_shape.packed());
    migraphx::gpu::gpu_fill(ctx, migraphx::argument{view_shape, buffer.data()}, 0);
    ctx.finish();

    // Only the view elements are zeroed; the rest of the buffer is untouched
    std::vector<float> expected = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    EXPECT(read_gpu(buffer) == expected);
}

TEST_CASE(fill_nonpacked_nonzero)
{
    // A non-zero fill on a non-packed tensor must assign whole elements, not bytes
    migraphx::gpu::context ctx{};
    migraphx::shape buffer_shape{migraphx::shape::float_type, {12}};
    std::vector<float> ones(buffer_shape.elements(), 1.0f);
    auto buffer = migraphx::gpu::to_gpu(migraphx::argument{buffer_shape, ones.data()});

    migraphx::shape view_shape{migraphx::shape::float_type, {6}, {2}};
    migraphx::gpu::gpu_fill(ctx, migraphx::argument{view_shape, buffer.data()}, 3);
    ctx.finish();

    // Every other element is assigned 3; the elements in between are untouched
    std::vector<float> expected = {3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1};
    EXPECT(read_gpu(buffer) == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
