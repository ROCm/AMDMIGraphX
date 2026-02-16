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
#include <migraphx/gpu/gen/fuse_gen.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape;

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::fuse_pointwise{},
                          migraphx::gpu::gen::fuse_gen{},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(fuse_add)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s   = shape{shape::float_type, {4, 8}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        mm->add_instruction(make_op("add"), x, y);
    }
    run_pass(p1);

    // After fuse_pointwise + fuse_gen, the add should be wrapped in gpu::gen::op
    auto* mm          = p1.get_main_module();
    bool found_gen_op = false;
    for(auto& ins : *mm)
    {
        if(ins.name() == "gpu::gen::op")
            found_gen_op = true;
    }
    EXPECT(found_gen_op);
}

TEST_CASE(fuse_mul_add)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s   = shape{shape::float_type, {16}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto mul = mm->add_instruction(make_op("mul"), x, y);
        mm->add_instruction(make_op("add"), mul, z);
    }
    run_pass(p1);

    auto* mm          = p1.get_main_module();
    bool found_gen_op = false;
    for(auto& ins : *mm)
    {
        if(ins.name() == "gpu::gen::op")
            found_gen_op = true;
    }
    EXPECT(found_gen_op);
}

TEST_CASE(no_fuse_identity)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s   = shape{shape::float_type, {4, 8}};
        auto x   = mm->add_parameter("x", s);
        mm->add_instruction(make_op("identity"), x);
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    // identity should not be fused into gen::op
    auto* mm          = p1.get_main_module();
    bool found_gen_op = false;
    for(auto& ins : *mm)
    {
        if(ins.name() == "gpu::gen::op")
            found_gen_op = true;
    }
    EXPECT(not found_gen_op);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
