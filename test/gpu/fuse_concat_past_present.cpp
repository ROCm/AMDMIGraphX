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
#include <migraphx/iterator_for.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <algorithm>
#include "make_precompile_op.hpp"

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::fuse_concat_past_present{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape cache_shape() { return {migraphx::shape::half_type, {1, 2, 8, 4}}; }

static migraphx::shape index_shape() { return {migraphx::shape::int32_type, {1, 1}}; }

static migraphx::shape present_shape(std::size_t seq)
{
    return {migraphx::shape::half_type, {1, 2, seq, 4}};
}

// The cache slot the fused producer writes into
static migraphx::shape slot_shape(std::size_t seq)
{
    return {migraphx::shape::half_type, present_shape(seq).lens(), cache_shape().strides()};
}

// pointwise(x, y) -> concat_past_present(pw, slk, cache), returning the concat
static migraphx::instruction_ref add_concat_past_present(migraphx::program& p, std::size_t seq)
{
    auto* mm   = p.get_main_module();
    auto s     = present_shape(seq);
    auto x     = mm->add_parameter("x", s);
    auto y     = mm->add_parameter("y", s);
    auto slk   = mm->add_parameter("slk", index_shape());
    auto cache = mm->add_parameter("cache", cache_shape());
    auto* pm   = create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("mul"));
    auto alloc =
        mm->add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto pw = mm->add_instruction(make_precompile_op("pointwise"), {x, y, alloc}, {pm});
    return mm->add_instruction(
        make_precompile_op(migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}),
                           cache_shape()),
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
        auto s     = present_shape(1);
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto slk    = mm->add_parameter("slk", index_shape());
        auto cache  = mm->add_parameter("cache", cache_shape());
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
        mm->add_parameter("slk", index_shape());
        auto cache = mm->add_parameter("cache", cache_shape());
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

// Unlowered decode append: mul(x, y) -> concat_past_present(mul, slk, cache)
static migraphx::program make_decode_program()
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto s     = present_shape(1);
    auto x     = mm->add_parameter("x", s);
    auto y     = mm->add_parameter("y", s);
    auto slk   = mm->add_parameter("slk", index_shape());
    auto cache = mm->add_parameter("cache", cache_shape());
    auto mul   = mm->add_instruction(migraphx::make_op("mul"), x, y);
    mm->add_return({mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), mul, slk, cache)});
    return p;
}

static bool has_fused_append(const migraphx::program& p)
{
    auto instructions = migraphx::iterator_for(*p.get_main_module());
    return std::any_of(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "hip::load_scalar";
    });
}

// The gpu target runs the pass by default and skips it when the
// eliminate_concat_past_present backend option is false.
static bool compiles_to_fused_append(bool enabled)
{
    auto p = make_decode_program();
    migraphx::compile_options options;
    options.backend_options["eliminate_concat_past_present"] = enabled;
    p.compile(migraphx::make_target("gpu"), options);
    return has_fused_append(p);
}

TEST_CASE(backend_option_enabled) { EXPECT(compiles_to_fused_append(true)); }

TEST_CASE(backend_option_disabled) { EXPECT(not compiles_to_fused_append(false)); }

// The fused append rejects a position outside the cache, on either bound,
// where the copy kernel would silently skip the write.
TEST_CASE(out_of_range_position_throws)
{
    auto p = make_decode_program();
    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), options);
    EXPECT(has_fused_append(p));

    auto append = [&](int pos) {
        migraphx::parameter_map params;
        params["x"]     = migraphx::generate_argument(present_shape(1), 1);
        params["y"]     = migraphx::generate_argument(present_shape(1), 2);
        params["cache"] = migraphx::generate_argument(cache_shape(), 3);
        params["slk"]   = migraphx::literal{index_shape(), {pos}}.get_argument();
        p.eval(params);
    };
    auto cache_len = cache_shape().lens()[2];
    append(0);
    append(cache_len - 1);
    EXPECT(test::throws([&] { append(-1); }));
    EXPECT(test::throws([&] { append(cache_len); }));
    // A rejected position leaves the program usable
    append(0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
