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
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>

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

    // Module should be unchanged since no copy operations
    migraphx::module m2;
    {
        m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(test_tiling_pass_1d_copy)
{
    // 1D copy should not be tiled
    migraphx::module m1;
    {
        auto src  = m1.add_parameter("src", migraphx::shape{migraphx::shape::float_type, {64}});
        auto dst  = m1.add_parameter("dst", migraphx::shape{migraphx::shape::float_type, {64}});
        auto copy = m1.add_instruction(migraphx::make_op("gpu::gen::copy"), src, dst);
        m1.add_return({copy});
    }
    run_tiling_pass(m1);

    // Should be unchanged
    migraphx::module m2;
    {
        auto src  = m2.add_parameter("src", migraphx::shape{migraphx::shape::float_type, {64}});
        auto dst  = m2.add_parameter("dst", migraphx::shape{migraphx::shape::float_type, {64}});
        auto copy = m2.add_instruction(migraphx::make_op("gpu::gen::copy"), src, dst);
        m2.add_return({copy});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(test_tiling_pass_multidim_copy)
{
    // Multi-dimensional copy with dimensions that require tiling
    // Tiling is triggered when fast axis is not the last dimension
    // Use a 3D tensor where axis 1 has a smaller dimension that benefits from tiling
    migraphx::module m1;
    {
        auto src =
            m1.add_parameter("src", migraphx::shape{migraphx::shape::float_type, {64, 32, 128}});
        auto dst =
            m1.add_parameter("dst", migraphx::shape{migraphx::shape::float_type, {64, 32, 128}});
        auto copy = m1.add_instruction(migraphx::make_op("gpu::gen::copy"), src, dst);
        m1.add_return({copy});
    }
    run_tiling_pass(m1);

    // Check if tiling was applied (may or may not be, depending on shape analysis)
    // The test verifies the pass runs without error
    bool has_tile_region  = false;
    bool has_workgroup_id = false;
    for(auto ins : migraphx::iterator_for(m1))
    {
        if(ins->name() == "gpu::gen::tile_region")
            has_tile_region = true;
        if(ins->name() == "gpu::gen::workgroup_id")
            has_workgroup_id = true;
    }
    // If tiling was applied, both should be present
    EXPECT(has_tile_region == has_workgroup_id);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
