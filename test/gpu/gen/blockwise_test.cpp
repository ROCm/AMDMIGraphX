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
#include <migraphx/gpu/gen/gridwise.hpp>
#include <migraphx/gpu/gen/blockwise.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape;

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::gpu::gen::gen_gridwise{},
                          migraphx::gpu::gen::gen_blockwise{},
                          migraphx::dead_code_elimination{}});
}

// 1D: no tiling, blockwise is a no-op. Gridwise adds z_output + copy.
TEST_CASE(blockwise_1d_no_tiling)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", shape{shape::float_type, {16}});
        auto y   = m1.add_parameter("y", shape{shape::float_type, {16}});
        auto add = m1.add_instruction(make_op("add"), x, y);
        m1.add_return({add});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", shape{shape::float_type, {16}});
        auto y   = m2.add_parameter("y", shape{shape::float_type, {16}});
        auto z   = m2.add_parameter("z_output", shape{shape::float_type, {16}});
        auto add = m2.add_instruction(make_op("add"), x, y);
        auto cp = m2.add_instruction(make_op("gpu::gen::copy", {{"schedule", "per_lane"}}), add, z);
        m2.add_return({cp});
    }

    EXPECT(m1.sort() == m2.sort());
}

// 2D tileable: blockwise inserts tile_region for each multi-dim param,
// including z_output. workgroup_id kept alive by tile_region refs.
TEST_CASE(blockwise_2d_tile_region)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", shape{shape::float_type, {16, 16}});
        auto y   = m1.add_parameter("y", shape{shape::float_type, {16, 16}});
        auto add = m1.add_instruction(make_op("add"), x, y);
        m1.add_return({add});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", shape{shape::float_type, {16, 16}});
        auto y    = m2.add_parameter("y", shape{shape::float_type, {16, 16}});
        auto z    = m2.add_parameter("z_output", shape{shape::float_type, {16, 16}});
        auto wgid = m2.add_instruction(make_op("gpu::gen::workgroup_id"));
        auto tx   = m2.add_instruction(
            make_op("gpu::gen::tile_region",
                      {{"tile_dims", std::vector<std::size_t>{16, 16}}, {"axis", 0}}),
            x,
            wgid);
        auto ty = m2.add_instruction(
            make_op("gpu::gen::tile_region",
                    {{"tile_dims", std::vector<std::size_t>{16, 16}}, {"axis", 0}}),
            y,
            wgid);
        auto tz = m2.add_instruction(
            make_op("gpu::gen::tile_region",
                    {{"tile_dims", std::vector<std::size_t>{16, 16}}, {"axis", 0}}),
            z,
            wgid);
        auto add = m2.add_instruction(make_op("add"), tx, ty);
        auto cp =
            m2.add_instruction(make_op("gpu::gen::copy", {{"schedule", "per_lane"}}), add, tz);
        m2.add_return({cp});
    }

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
