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
#include <migraphx/compile_options.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <test.hpp>

using migraphx::shape;
using lens_t = std::vector<std::size_t>;

enum class post_op
{
    none,
    bias_silu,
    bias_relu,
    residual_add
};

static bool is_gfx12()
{
    auto ctx = migraphx::gpu::context{};
    return migraphx::starts_with(ctx.get_current_device().get_gfx_name(), "gfx12");
}

static migraphx::program make_program(const lens_t& x_lens,
                                      std::size_t k,
                                      post_op post                      = post_op::bias_silu,
                                      const migraphx::value& conv_attrs = {},
                                      shape::type_t type                = shape::float_type,
                                      std::size_t group                 = 1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {type, x_lens});
    auto w   = mm->add_literal(migraphx::generate_literal({type, {k, x_lens[1] / group, 3, 3}}, 4));
    auto attrs = migraphx::value{{"padding", {1, 1}}, {"stride", {2, 2}}, {"group", group}};
    for(const auto& a : conv_attrs)
        attrs[a.get_key()] = a.without_key();
    auto y               = mm->add_instruction(migraphx::make_op("convolution", attrs), x, w);
    const auto& out_lens = y->get_shape().lens();
    if(post == post_op::bias_silu or post == post_op::bias_relu)
    {
        auto b  = mm->add_literal(migraphx::generate_literal({type, {k}}, 5));
        auto bb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", out_lens}}), b);
        y = mm->add_instruction(migraphx::make_op("add"), y, bb);
        if(post == post_op::bias_silu)
            y = mm->add_instruction(
                migraphx::make_op("mul"), y, mm->add_instruction(migraphx::make_op("sigmoid"), y));
        else
            y = mm->add_instruction(migraphx::make_op("relu"), y);
    }
    else if(post == post_op::residual_add)
    {
        auto r = mm->add_parameter("r", {type, out_lens});
        y      = mm->add_instruction(migraphx::make_op("add"), y, r);
    }
    mm->add_return({y});
    return p;
}

static std::vector<float> run(migraphx::program p,
                              const std::string& target,
                              const migraphx::parameter_map& params,
                              bool* uses_kernel = nullptr)
{
    migraphx::compile_options options;
    options.offload_copy = true;
    if(target == "gpu")
        migraphx::set_backend_options(options, {{"convolution_layout", "channels_first"}});
    p.compile(migraphx::make_target(target), options);
    if(uses_kernel != nullptr)
        *uses_kernel = migraphx::contains(migraphx::to_string(p), "symbol_name=conv3x3s2");
    auto result = p.eval(params).back();
    std::vector<float> out;
    result.visit([&](auto v) { out.assign(v.begin(), v.end()); });
    return out;
}

// Compiles p for gpu and ref, checks the results agree and returns whether gpu used conv3x3s2.
static bool check(const migraphx::program& p)
{
    migraphx::parameter_map params;
    std::size_t seed = 6;
    for(const auto& [name, s] : p.get_parameter_shapes())
        params[name] = migraphx::generate_argument(s, seed++);
    bool uses_kernel = false;
    auto gpu         = run(p, "gpu", params, &uses_kernel);
    auto ref         = run(p, "ref", params);
    EXPECT(migraphx::verify::verify_rms_range(gpu, ref));
    return uses_kernel;
}

// Edge cases: partial pixel groups (odd output width), output channels that are not a
// multiple of 16 or fewer than one channel tile, one input channel per wave, batch > 1,
// a single output pixel, and tall/narrow and wide/short planes.
TEST_CASE(conv3x3s2_fp32_shapes)
{
    if(not is_gfx12())
        return;
    const std::vector<std::pair<lens_t, std::size_t>> cases = {{{1, 24, 30, 22}, 20},
                                                               {{2, 8, 16, 16}, 16},
                                                               {{1, 17, 10, 14}, 5},
                                                               {{1, 32, 2, 2}, 8},
                                                               {{1, 8, 64, 6}, 40},
                                                               {{3, 16, 12, 20}, 33},
                                                               {{1, 64, 4, 70}, 24},
                                                               {{1, 8, 8, 8}, 1}};
    for(const auto& [x_lens, k] : cases)
    {
        EXPECT(check(make_program(x_lens, k)));
    }
}

TEST_CASE(conv3x3s2_fp32_fused_post_ops)
{
    if(not is_gfx12())
        return;
    for(auto post : {post_op::none, post_op::bias_relu, post_op::residual_add})
    {
        EXPECT(check(make_program({2, 12, 18, 26}, 20, post)));
    }
}

// Whether prefuse_ops rewrites the convolution in p to gpu::conv3x3s2.
static bool rewritten(migraphx::program p)
{
    migraphx::gpu::context ctx;
    auto* mm = p.get_main_module();
    migraphx::run_passes(*mm, {migraphx::gpu::prefuse_ops{.ctx = &ctx}});
    return std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gpu::conv3x3s2"; });
}

// Convolutions outside the kernel's contract keep the default path.
TEST_CASE(conv3x3s2_matcher)
{
    if(not is_gfx12())
        return;
    const lens_t x{1, 8, 16, 16};
    EXPECT(rewritten(make_program(x, 8)));
    EXPECT(not rewritten(make_program({1, 8, 15, 16}, 8)));
    EXPECT(not rewritten(make_program({1, 8, 16, 15}, 8)));
    EXPECT(not rewritten(make_program({1, 3, 16, 16}, 8)));
    EXPECT(not rewritten(make_program(x, 8, post_op::bias_silu, {{"stride", {1, 1}}})));
    EXPECT(not rewritten(make_program(x, 8, post_op::bias_silu, {{"stride", {2, 1}}})));
    EXPECT(not rewritten(make_program(x, 8, post_op::bias_silu, {{"padding", {0, 0}}})));
    EXPECT(not rewritten(make_program(x, 8, post_op::bias_silu, {{"dilation", {2, 2}}})));
    EXPECT(not rewritten(make_program(x, 8, post_op::bias_silu, {}, shape::float_type, 2)));
    EXPECT(not rewritten(make_program(x, 8, post_op::bias_silu, {}, shape::half_type)));

    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}),
                        mm->add_parameter("x", {shape::float_type, x}),
                        mm->add_parameter("w", {shape::float_type, {8, 8, 3, 3}}));
    EXPECT(not rewritten(p));
}

// [K, C, 3, 3] -> [C, 9, KPAD], KPAD = K rounded up to 16, as the prefuse_ops rewrite does.
static migraphx::literal transpose_weights(const migraphx::literal& w)
{
    const auto& l  = w.get_shape().lens();
    std::size_t kp = (l[0] + 15) / 16 * 16;
    std::vector<float> wt(l[1] * 9 * kp, 0.0f);
    auto v = w.to_vector<float>();
    for(std::size_t k = 0; k < l[0]; ++k)
        for(std::size_t c = 0; c < l[1]; ++c)
            for(std::size_t t = 0; t < 9; ++t)
                wt[(c * 9 + t) * kp + k] = v[(k * l[1] + c) * 9 + t];
    return migraphx::literal{shape{shape::float_type, {l[1], 9, kp}}, wt};
}

// Every tuning candidate, compiled directly, matches the reference convolution.
TEST_CASE(conv3x3s2_fp32_all_tiles)
{
    if(not is_gfx12())
        return;
    const std::vector<std::pair<lens_t, std::size_t>> cases = {
        {{1, 3, 6, 10}, 5}, {{2, 17, 10, 14}, 20}, {{1, 24, 30, 22}, 33}};
    std::vector<migraphx::value> tiles;
    for(std::size_t tp : {1, 2, 4})
        for(std::size_t nw : {1, 2, 4, 8})
            tiles.push_back({{"tk", 8}, {"tp", tp}, {"nw", nw}});
    tiles.push_back({{"tk", 16}, {"tp", 2}, {"nw", 1}});
    tiles.push_back({{"tk", 16}, {"tp", 2}, {"nw", 4}});

    migraphx::gpu::context ctx;
    for(const auto& [x_lens, k] : cases)
    {
        shape x_s{shape::float_type, x_lens};
        shape out_s{shape::float_type, {x_lens[0], k, x_lens[2] / 2, x_lens[3] / 2}};
        auto x = migraphx::generate_literal(x_s, 1);
        auto w = migraphx::generate_literal({shape::float_type, {k, x_lens[1], 3, 3}}, 2);

        migraphx::program rp;
        auto* rm = rp.get_main_module();
        rm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}}),
            rm->add_literal(x),
            rm->add_literal(w));
        rp.compile(migraphx::make_target("ref"));
        std::vector<float> ref;
        rp.eval({}).back().visit([&](auto v) { ref.assign(v.begin(), v.end()); });

        auto wt = transpose_weights(w);
        for(const auto& tile : tiles)
        {
            auto co =
                migraphx::gpu::compile_op("gpu::conv3x3s2", ctx, {x_s, wt.get_shape(), out_s}, tile);
            migraphx::program p;
            auto* mm = p.get_main_module();
            mm->add_instruction(co,
                                mm->add_literal(x),
                                mm->add_literal(wt),
                                mm->add_parameter("output", out_s));
            p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});
            auto result = migraphx::gpu::from_gpu(
                p.eval({{"output", migraphx::gpu::allocate_gpu(out_s)}}).front());
            std::vector<float> gpu;
            result.visit([&](auto v) { gpu.assign(v.begin(), v.end()); });
            EXPECT(migraphx::verify::verify_rms_range(gpu, ref));
        }
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
