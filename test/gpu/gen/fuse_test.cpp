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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <pointwise.hpp>

using migraphx::make_op;
using migraphx::shape;

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::gen::fuse_gen{}, migraphx::dead_code_elimination{}});
}

template <class F>
migraphx::instruction_ref add_gen_op(migraphx::program& p,
                                     const std::string& name,
                                     std::vector<migraphx::instruction_ref> inputs,
                                     const migraphx::operation& orig_op,
                                     const F& f)
{
    auto* mm = p.get_main_module();
    auto* pm = p.create_module(name);
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    for(std::size_t i = 0; i < inputs.size(); ++i)
    {
        params.push_back(
            pm->add_parameter("x" + std::to_string(i), inputs[i]->get_shape()));
    }
    auto r = f(pm, params);
    pm->add_return({r});
    (void)orig_op;
    return mm->add_instruction(make_op("gpu::gen::op"), inputs, {pm});
}

TEST_CASE(fuse_pointwise_add)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s   = shape{shape::float_type, {4, 8}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s   = shape{shape::float_type, {4, 8}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        add_gen_op(
            p2,
            "gen_main:pointwise0",
            {x, y},
            make_op("pointwise"),
            [](auto* pm, const auto& params) {
                return pm->add_instruction(migraphx::make_op("add"), params[0], params[1]);
            });
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fuse_pointwise_mul_add)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s   = shape{shape::float_type, {16}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        add_pointwise(p1, "main:pointwise0", {x, y, z}, [](auto* pm, const auto& params) {
            auto mul = pm->add_instruction(migraphx::make_op("mul"), params[0], params[1]);
            return pm->add_instruction(migraphx::make_op("add"), mul, params[2]);
        });
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s   = shape{shape::float_type, {16}};
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        add_gen_op(
            p2,
            "gen_main:pointwise0",
            {x, y, z},
            make_op("pointwise"),
            [](auto* pm, const auto& params) {
                auto mul =
                    pm->add_instruction(migraphx::make_op("mul"), params[0], params[1]);
                return pm->add_instruction(migraphx::make_op("add"), mul, params[2]);
            });
    }
    EXPECT(p1.sort() == p2.sort());
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

    // identity should not be fused
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
