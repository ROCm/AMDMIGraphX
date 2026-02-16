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
#include <migraphx/gpu/gen/lanewise.hpp>
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
                          migraphx::gpu::gen::gen_lanewise{},
                          migraphx::dead_code_elimination{}});
}

// 1D pointwise: gridwise adds z_output + copy, lanewise lowers to
// load/add/store with global_id
TEST_CASE(lanewise_1d_pointwise)
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
        auto gid = m2.add_instruction(make_op("gpu::gen::global_id"));
        auto lx  = m2.add_instruction(make_op("gpu::gen::load"), x, gid);
        auto ly  = m2.add_instruction(make_op("gpu::gen::load"), y, gid);
        auto add = m2.add_instruction(make_op("add"), lx, ly);
        auto st  = m2.add_instruction(make_op("gpu::gen::store"), z, gid, add);
        m2.add_return({st});
    }

    EXPECT(m1.sort() == m2.sort());
}

// 2D tiled with copy: local_id used, tile_region present
TEST_CASE(lanewise_2d_copy_uses_local_id)
{
    shape s{shape::float_type, {16, 16}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto add = m1.add_instruction(make_op("add"), x, y);
        m1.add_return({add});
    }
    run_pass(m1);

    // After full pipeline on 2D: should have local_id (from tiling)
    bool found_local_id = false;
    for(auto& ins : m1)
    {
        if(ins.name() == "gpu::gen::local_id")
            found_local_id = true;
    }
    EXPECT(found_local_id);
}

// Block reduce (256 elements, 256 block_size -> 1 element per thread):
// load + dpp_reduce + reduce_waves (no lane_reduce since each thread has 1 element)
TEST_CASE(lanewise_block_reduce_small)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", shape{shape::float_type, {256}});
        auto red = m1.add_instruction(make_op("reduce_sum", {{"axes", {0}}}), x);
        m1.add_return({red});
    }
    run_pass(m1);

    bool found_dpp_reduce   = false;
    bool found_reduce_waves = false;
    for(auto& ins : m1)
    {
        if(ins.name() == "gpu::gen::dpp_reduce")
            found_dpp_reduce = true;
        if(ins.name() == "gpu::gen::reduce_waves")
            found_reduce_waves = true;
    }
    EXPECT(found_dpp_reduce);
    EXPECT(found_reduce_waves);
}

// Block reduce (1024 elements -> multiple elements per thread):
// strided_load + lane_reduce + dpp_reduce + reduce_waves
TEST_CASE(lanewise_block_reduce_large)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", shape{shape::float_type, {1024}});
        auto red = m1.add_instruction(make_op("reduce_sum", {{"axes", {0}}}), x);
        m1.add_return({red});
    }
    run_pass(m1);

    bool found_lane_reduce  = false;
    bool found_dpp_reduce   = false;
    bool found_reduce_waves = false;
    for(auto& ins : m1)
    {
        if(ins.name() == "gpu::gen::lane_reduce")
            found_lane_reduce = true;
        if(ins.name() == "gpu::gen::dpp_reduce")
            found_dpp_reduce = true;
        if(ins.name() == "gpu::gen::reduce_waves")
            found_reduce_waves = true;
    }
    EXPECT(found_lane_reduce);
    EXPECT(found_dpp_reduce);
    EXPECT(found_reduce_waves);
}

// Wave reduce: gridwise_reduce[algo=wave] -> load + dpp_reduce (no reduce_waves)
TEST_CASE(lanewise_wave_reduce)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", shape{shape::float_type, {32}});
        auto red = m1.add_instruction(make_op("reduce_sum", {{"axes", {0}}}), x);
        m1.add_return({red});
    }
    run_pass(m1);

    bool found_dpp_reduce   = false;
    bool found_reduce_waves = false;
    for(auto& ins : m1)
    {
        if(ins.name() == "gpu::gen::dpp_reduce")
            found_dpp_reduce = true;
        if(ins.name() == "gpu::gen::reduce_waves")
            found_reduce_waves = true;
    }
    EXPECT(found_dpp_reduce);
    EXPECT(not found_reduce_waves);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
