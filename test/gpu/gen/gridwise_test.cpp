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
                         {migraphx::gpu::gen::gen_gridwise{}, migraphx::dead_code_elimination{}});
}

// No @return in module: pass is a no-op
TEST_CASE(gridwise_no_return_no_change)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", shape{shape::float_type, {64, 128}});
        auto y = m1.add_parameter("y", shape{shape::float_type, {64, 128}});
        m1.add_instruction(make_op("add"), x, y);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", shape{shape::float_type, {64, 128}});
        auto y = m2.add_parameter("y", shape{shape::float_type, {64, 128}});
        m2.add_instruction(make_op("add"), x, y);
    }

    EXPECT(m1 == m2);
}

// 1D: gridwise adds z_output param and copy, no workgroup_id (1D has ntiles=0)
TEST_CASE(gridwise_1d_adds_output_and_copy)
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

// 2D tileable: gridwise adds z_output, copy, AND workgroup_id
// DCE removes workgroup_id because nothing uses it at gridwise level
TEST_CASE(gridwise_2d_adds_output_copy_wgid_removed)
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
        auto x   = m2.add_parameter("x", shape{shape::float_type, {16, 16}});
        auto y   = m2.add_parameter("y", shape{shape::float_type, {16, 16}});
        auto z   = m2.add_parameter("z_output", shape{shape::float_type, {16, 16}});
        auto add = m2.add_instruction(make_op("add"), x, y);
        auto cp = m2.add_instruction(make_op("gpu::gen::copy", {{"schedule", "per_lane"}}), add, z);
        m2.add_return({cp});
    }

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
