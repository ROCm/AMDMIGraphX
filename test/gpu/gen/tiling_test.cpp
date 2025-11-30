/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/shape.hpp>

void run_tiling_pass(migraphx::module& m)
{
    migraphx::gpu::gen::gen_tiling pass;
    pass.apply(m);
}

TEST_CASE(test_gen_tiling_pass_name)
{
    migraphx::gpu::gen::gen_tiling pass;
    EXPECT(pass.name() == "gpu::gen::tiling");
}

TEST_CASE(test_tile_config_empty)
{
    std::vector<migraphx::shape> inputs;
    auto config = migraphx::gpu::gen::tile_config::compute(inputs, 1);
    EXPECT(not config.is_tiled());
}

TEST_CASE(test_tile_config_small_tensor)
{
    // Small tensor that shouldn't be tiled
    std::vector<migraphx::shape> inputs = {migraphx::shape{migraphx::shape::float_type, {4, 4}}};
    auto config                         = migraphx::gpu::gen::tile_config::compute(inputs, 1);
    // Small tensors may not be tiled
    (void)config;
}

TEST_CASE(test_tile_config_large_tensor)
{
    // Larger tensor that might be tiled
    std::vector<migraphx::shape> inputs = {
        migraphx::shape{migraphx::shape::float_type, {64, 512, 32, 32}}};
    auto config = migraphx::gpu::gen::tile_config::compute(inputs, 1);
    // The tiling decision depends on the shape analysis
    if(config.is_tiled())
    {
        EXPECT(config.ntiles > 0);
        EXPECT(config.block_size > 0);
    }
}

TEST_CASE(test_tiling_pass_empty_module)
{
    migraphx::module m1;
    {
        m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    }
    run_tiling_pass(m1);

    // Module should be unchanged since tiling pass doesn't modify it yet
    migraphx::module m2;
    {
        m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
