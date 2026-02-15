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
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/shape.hpp>
#include <test.hpp>

using migraphx::shape;
using migraphx::gpu::gen::compute_tile_config;
using migraphx::gpu::gen::compute_vec_size;

TEST_CASE(vec_size_float_aligned)
{
    std::vector<shape> inputs = {
        shape{shape::float_type, {256}},
        shape{shape::float_type, {256}}};
    auto vec = compute_vec_size(inputs);
    EXPECT(vec == 4);
}

TEST_CASE(vec_size_float_unaligned)
{
    std::vector<shape> inputs = {
        shape{shape::float_type, {3}},
        shape{shape::float_type, {3}}};
    auto vec = compute_vec_size(inputs);
    EXPECT(vec == 1);
}

TEST_CASE(vec_size_half)
{
    std::vector<shape> inputs = {
        shape{shape::half_type, {256}},
        shape{shape::half_type, {256}}};
    auto vec = compute_vec_size(inputs);
    EXPECT(vec == 2);
}

TEST_CASE(tile_config_1d)
{
    std::vector<shape> inputs = {shape{shape::float_type, {1024}}};
    auto config               = compute_tile_config(inputs);
    EXPECT(config.block_size == 256);
    EXPECT(config.grid_size > 0);
    EXPECT(config.tile_dims.empty());
}

TEST_CASE(tile_config_2d)
{
    std::vector<shape> inputs = {shape{shape::float_type, {64, 128}}};
    auto config               = compute_tile_config(inputs);
    EXPECT(config.block_size > 0);
    EXPECT(config.grid_size > 0);
}

TEST_CASE(tile_config_empty)
{
    std::vector<shape> inputs = {};
    auto config               = compute_tile_config(inputs);
    EXPECT(config.block_size == 256);
    EXPECT(config.vec_size == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
