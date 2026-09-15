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
#include <migraphx/gpu/fuse_concat_past_present.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::fuse_concat_past_present{}, migraphx::dead_code_elimination{}});
}

static migraphx::operation precompile(const migraphx::operation& op)
{
    return migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(op)}});
}

static migraphx::operation precompile(const migraphx::operation& op, const migraphx::shape& s)
{
    return migraphx::make_op(
        "gpu::precompile_op",
        {{"op", migraphx::to_value(op)}, {"output_shape", migraphx::to_value(s)}});
}

TEST_CASE(fuse_decode)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 2, 1, 4}};
    migraphx::shape cs{migraphx::shape::half_type, {1, 2, 8, 4}};
    migraphx::shape is{migraphx::shape::int32_type, {1, 1}};
    migraphx::shape vs{migraphx::shape::half_type, {1, 2, 1, 4}, {64, 32, 4, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto slk   = mm->add_parameter("slk", is);
        auto cache = mm->add_parameter("cache", cs);
        auto* pm = create_pointwise_module(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto pw =
            mm->add_instruction(precompile(migraphx::make_op("pointwise")), {x, y, alloc}, {pm});
        auto cpp = mm->add_instruction(
            precompile(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), cs),
            pw,
            slk,
            cache);
        mm->add_return({cpp});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto slk   = mm->add_parameter("slk", is);
        auto cache = mm->add_parameter("cache", cs);
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto scalar = mm->add_instruction(migraphx::make_op("gpu::load_scalar"), slk);
        auto view =
            mm->add_instruction(migraphx::make_op("gpu::slice_at", {{"axis", 2}}), cache, scalar);
        auto pw =
            mm->add_instruction(precompile(migraphx::make_op("pointwise"), vs), {x, y, view}, {pm});
        auto dep = mm->add_instruction(migraphx::make_op("gpu::depends_on"), cache, pw);
        mm->add_return({dep});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fuse_prefill)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 2, 4, 4}};
    migraphx::shape cs{migraphx::shape::half_type, {1, 2, 8, 4}};
    migraphx::shape is{migraphx::shape::int32_type, {1, 1}};
    migraphx::shape vs{migraphx::shape::half_type, {1, 2, 4, 4}, {64, 32, 4, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto slk   = mm->add_parameter("slk", is);
        auto cache = mm->add_parameter("cache", cs);
        auto* pm = create_pointwise_module(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto pw =
            mm->add_instruction(precompile(migraphx::make_op("pointwise")), {x, y, alloc}, {pm});
        auto cpp = mm->add_instruction(
            precompile(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), cs),
            pw,
            slk,
            cache);
        mm->add_return({cpp});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        mm->add_parameter("slk", is);
        auto cache = mm->add_parameter("cache", cs);
        auto* pm  = create_pointwise_module(p2, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto view = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), cache);
        auto pw =
            mm->add_instruction(precompile(migraphx::make_op("pointwise"), vs), {x, y, view}, {pm});
        auto dep = mm->add_instruction(migraphx::make_op("gpu::depends_on"), cache, pw);
        mm->add_return({dep});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(skip_multi_use_producer)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 2, 1, 4}};
    migraphx::shape cs{migraphx::shape::half_type, {1, 2, 8, 4}};
    migraphx::shape is{migraphx::shape::int32_type, {1, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto slk   = mm->add_parameter("slk", is);
        auto cache = mm->add_parameter("cache", cs);
        auto* pm = create_pointwise_module(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto pw =
            mm->add_instruction(precompile(migraphx::make_op("pointwise")), {x, y, alloc}, {pm});
        auto cpp = mm->add_instruction(
            precompile(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), cs),
            pw,
            slk,
            cache);
        mm->add_return({cpp, pw});
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
