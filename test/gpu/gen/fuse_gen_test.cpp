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
#include <migraphx/gpu/gen/fuse_gen.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>

void run_fuse_gen_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::gen::fuse_gen{}});
}

TEST_CASE(test_fuse_gen_pass_name)
{
    migraphx::gpu::gen::fuse_gen pass;
    EXPECT(pass.name() == "gpu::gen::fuse_gen");
}

TEST_CASE(test_fuse_gen_pass_disabled_by_default)
{
    // fuse_gen should not modify the module when MIGRAPHX_ENABLE_GEN is not set
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto y = m1.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto add_op = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({add_op});
    }
    
    // Expected: module unchanged since MIGRAPHX_ENABLE_GEN is not set
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto y = m2.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto add_op = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({add_op});
    }

    run_fuse_gen_pass(m1);
    
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(test_gen_pointwise_compute_shape)
{
    // Create a simple pointwise module
    migraphx::module pm;
    {
        auto px = pm.add_parameter("x0", migraphx::shape{migraphx::shape::float_type});
        auto py = pm.add_parameter("x1", migraphx::shape{migraphx::shape::float_type});
        auto padd = pm.add_instruction(migraphx::make_op("add"), px, py);
        pm.add_return({padd});
    }

    // Test that gen::pointwise operation can be created
    auto gen_pw_op = migraphx::make_op("gpu::gen::pointwise");
    EXPECT(gen_pw_op.name() == "gpu::gen::pointwise");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
