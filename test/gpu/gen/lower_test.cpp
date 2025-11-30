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
#include <migraphx/gpu/gen/lower.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>

void run_lower_pass(migraphx::module& m)
{
    migraphx::gpu::gen::gen_lower pass;
    pass.apply(m);
    // Run DCE to remove unused instructions
    migraphx::dead_code_elimination{}.apply(m);
}

TEST_CASE(test_gen_lower_pass_name)
{
    migraphx::gpu::gen::gen_lower pass;
    EXPECT(pass.name() == "gpu::gen::lower");
}

TEST_CASE(test_gen_lower_empty_module)
{
    migraphx::module m1;
    {
        m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    }
    run_lower_pass(m1);
    
    migraphx::module m2;
    {
        m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(test_gen_lower_with_tile_region)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {64, 64}});
        auto tile_op = migraphx::make_op("gpu::gen::tile_region",
                                          {{"tile_dims", std::vector<std::size_t>{8, 8}},
                                           {"axis", std::size_t{0}}});
        auto tiled = m1.add_instruction(tile_op, x);
        m1.add_return({tiled});
    }
    run_lower_pass(m1);

    // For now, lower pass doesn't transform tile_region yet
    // So expected output should be unchanged
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {64, 64}});
        auto tile_op = migraphx::make_op("gpu::gen::tile_region",
                                          {{"tile_dims", std::vector<std::size_t>{8, 8}},
                                           {"axis", std::size_t{0}}});
        auto tiled = m2.add_instruction(tile_op, x);
        m2.add_return({tiled});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(test_gen_lower_copy)
{
    // Test that copy operation gets lowered to vector_load/vector_store
    migraphx::module m1;
    {
        auto src = m1.add_parameter("src", migraphx::shape{migraphx::shape::float_type, {64}});
        auto dst = m1.add_parameter("dst", migraphx::shape{migraphx::shape::float_type, {64}});
        auto copy = m1.add_instruction(migraphx::make_op("gpu::gen::copy"), src, dst);
        m1.add_return({copy});
    }
    run_lower_pass(m1);

    // After lowering, copy should become: global_id -> vector_load -> vector_store
    // Vector size is 8 because 64 % 8 == 0
    migraphx::module m2;
    {
        auto src = m2.add_parameter("src", migraphx::shape{migraphx::shape::float_type, {64}});
        auto dst = m2.add_parameter("dst", migraphx::shape{migraphx::shape::float_type, {64}});
        auto gid = m2.add_instruction(migraphx::make_op("gpu::gen::global_id"));
        auto load = m2.add_instruction(
            migraphx::make_op("gpu::gen::vector_load", {{"size", std::size_t{8}}}), src, gid);
        auto store = m2.add_instruction(
            migraphx::make_op("gpu::gen::vector_store", {{"size", std::size_t{8}}}), dst, gid, load);
        m2.add_return({store});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
