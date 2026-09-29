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
#include "make_precompile_op.hpp"

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::fuse_concat_past_present{}, migraphx::dead_code_elimination{}});
}

static const migraphx::shape cache_shape{migraphx::shape::half_type, {1, 2, 8, 4}};
static const migraphx::shape index_shape{migraphx::shape::int32_type, {1, 1}};

static migraphx::shape present_shape(std::size_t seq)
{
    return {migraphx::shape::half_type, {1, 2, seq, 4}};
}

// The cache slot the fused producer writes into
static migraphx::shape slot_shape(std::size_t seq)
{
    return {migraphx::shape::half_type, present_shape(seq).lens(), cache_shape.strides()};
}

// pointwise(x, y) -> concat_past_present(pw, slk, cache), returning the concat
static migraphx::instruction_ref add_concat_past_present(migraphx::program& p, std::size_t seq)
{
    auto* mm   = p.get_main_module();
    auto s     = present_shape(seq);
    auto x     = mm->add_parameter("x", s);
    auto y     = mm->add_parameter("y", s);
    auto slk   = mm->add_parameter("slk", index_shape);
    auto cache = mm->add_parameter("cache", cache_shape);
    auto* pm   = create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("mul"));
    auto alloc =
        mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto pw = mm->add_instruction(make_precompile_op("pointwise"), {x, y, alloc}, {pm});
    return mm->add_instruction(
        make_precompile_op(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}),
                           cache_shape),
        pw,
        slk,
        cache);
}

TEST_CASE(fuse_decode)
{
    migraphx::program p1;
    {
        auto cpp = add_concat_past_present(p1, 1);
        p1.get_main_module()->add_return({cpp});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto s      = present_shape(1);
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto slk    = mm->add_parameter("slk", index_shape);
        auto cache  = mm->add_parameter("cache", cache_shape);
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto scalar = mm->add_instruction(migraphx::make_op("hip::load_scalar"), slk);
        auto view =
            mm->add_instruction(migraphx::make_op("gpu::slice_at", {{"axis", 2}}), cache, scalar);
        auto pw = mm->add_instruction(
            make_precompile_op(migraphx::make_op("pointwise"), slot_shape(1)), {x, y, view}, {pm});
        auto dep = mm->add_instruction(migraphx::make_op("identity"), cache, pw);
        mm->add_return({dep});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fuse_prefill)
{
    migraphx::program p1;
    {
        auto cpp = add_concat_past_present(p1, 4);
        p1.get_main_module()->add_return({cpp});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s   = present_shape(4);
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        mm->add_parameter("slk", index_shape);
        auto cache = mm->add_parameter("cache", cache_shape);
        auto* pm  = create_pointwise_module(p2, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto view = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), cache);
        auto pw = mm->add_instruction(
            make_precompile_op(migraphx::make_op("pointwise"), slot_shape(4)), {x, y, view}, {pm});
        auto dep = mm->add_instruction(migraphx::make_op("identity"), cache, pw);
        mm->add_return({dep});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(skip_multi_use_producer)
{
    migraphx::program p1;
    {
        auto cpp = add_concat_past_present(p1, 1);
        p1.get_main_module()->add_return({cpp, cpp->inputs().front()});
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
