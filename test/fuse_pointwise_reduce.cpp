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
#include <migraphx/fuse_pointwise_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <test.hpp>

static void run_pass(migraphx::program& p, migraphx::fuse_pointwise_reduce pass = {})
{
    migraphx::run_passes(p, {pass, migraphx::dead_code_elimination{}});
}

static migraphx::program two_op_pointwise_program()
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", s);
    auto y    = mm->add_parameter("y", s);
    auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto mul1 = mm->add_instruction(migraphx::make_op("mul"), add1, x);
    mm->add_return({mul1});
    return p;
}

static std::size_t count_op(const migraphx::module& m, const std::string& name)
{
    return std::count_if(m.begin(), m.end(), [&](const auto& ins) { return ins.name() == name; });
}

// run_optimize_module gates fuse_pointwise_reduce's internal optimize_module{} run, which
// gpu/target.cpp turns off in eager compile mode. Either way the pass must still collapse a
// pointwise chain into one pointwise module.
TEST_CASE(fuses_pointwise_chain_with_optimize_module)
{
    auto p = two_op_pointwise_program();
    run_pass(p, migraphx::fuse_pointwise_reduce{.run_optimize_module = true});
    const auto& m = *p.get_main_module();
    EXPECT(count_op(m, "pointwise") == 1);
    EXPECT(count_op(m, "add") == 0);
    EXPECT(count_op(m, "mul") == 0);
}

TEST_CASE(fuses_pointwise_chain_without_optimize_module)
{
    auto p = two_op_pointwise_program();
    run_pass(p, migraphx::fuse_pointwise_reduce{.run_optimize_module = false});
    const auto& m = *p.get_main_module();
    EXPECT(count_op(m, "pointwise") == 1);
    EXPECT(count_op(m, "add") == 0);
    EXPECT(count_op(m, "mul") == 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
