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
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/value.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <algorithm>
#include <test.hpp>
#include "make_precompile_op.hpp"

// Compiles the channelwise conv kernel directly with a chosen tuning solution and layout
// and checks the result against the reference target. Going through the target's passes
// would let layout_convolution pick the layout and the tuner pick the solution.

struct channelwise_gpu_target : migraphx::gpu::target
{
    std::string name() const { return "channelwise"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {migraphx::gpu::write_literals{}};
    }
};

static migraphx::shape conv_output(const migraphx::shape& x_shape,
                                   const migraphx::shape& w_shape,
                                   const std::vector<std::size_t>& padding)
{
    auto lens = x_shape.lens();
    lens[1]   = w_shape.lens()[0];
    for(std::size_t i = 2; i < lens.size(); i++)
        lens[i] =
            lens[i] + padding[i - 2] + padding[i - 2 + lens.size() - 2] - w_shape.lens()[i] + 1;
    return x_shape.with_lens(lens);
}

static migraphx::instruction_ref add_ref_conv(migraphx::module& mm,
                                              migraphx::instruction_ref x,
                                              migraphx::instruction_ref w,
                                              const std::vector<std::size_t>& padding)
{
    std::vector<std::size_t> ones(x->get_shape().ndim() - 2, 1);
    return mm.add_instruction(migraphx::make_op("convolution",
                                                {{"group", x->get_shape().lens()[1]},
                                                 {"padding", padding},
                                                 {"stride", ones},
                                                 {"dilation", ones}}),
                              x,
                              w);
}

static migraphx::argument run_ref(const migraphx::shape& x_shape,
                                  const migraphx::shape& w_shape,
                                  const std::vector<std::size_t>& padding,
                                  const migraphx::parameter_map& inputs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", x_shape);
    auto w   = mm->add_parameter("w", w_shape);
    add_ref_conv(*mm, x, w, padding);
    p.compile(migraphx::make_target("ref"));
    return p.eval(inputs).front();
}

// Run the kernel with `solution` and a fused pointwise module built by `build_post` over
// the conv result and the extra inputs, mirroring find_channelwise_conv_pointwise
template <class BuildPost>
static migraphx::argument run_gpu_fused(const std::vector<migraphx::shape>& param_shapes,
                                        const migraphx::shape& out_shape,
                                        const std::vector<std::size_t>& padding,
                                        const migraphx::parameter_map& inputs,
                                        const migraphx::value& solution,
                                        BuildPost build_post)
{
    const auto& x_shape = param_shapes.front();
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::instruction_ref> args;
    std::transform(param_shapes.begin(),
                   param_shapes.end(),
                   std::back_inserter(args),
                   [&](const migraphx::shape& s) {
                       return mm->add_parameter("p" + std::to_string(args.size()), s);
                   });
    args.push_back(mm->add_parameter("output", out_shape));

    auto* pm = p.create_module("post");
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    params.push_back(pm->add_parameter("x0", migraphx::shape{x_shape.type()}));
    std::transform(param_shapes.begin() + 2,
                   param_shapes.end(),
                   std::back_inserter(params),
                   [&](const migraphx::shape& s) {
                       return pm->add_parameter("x" + std::to_string(params.size()),
                                                migraphx::shape{s.type()});
                   });
    pm->add_return({build_post(*pm, params)});

    auto cw_op = migraphx::make_op("gpu::channelwise_conv",
                                   {{"num_spatial", x_shape.ndim() - 2}, {"padding", padding}});
    auto pre   = migraphx::make_op("gpu::precompile_op",
                                 {{"op", migraphx::to_value(cw_op)},
                                  {"output_shape", migraphx::to_value(out_shape)}});
    auto ins   = mm->add_instruction(pre, args, {pm});
    mm->add_return({ins});

    migraphx::gpu::context ctx;
    auto cr = migraphx::gpu::compile(ctx, ins, cw_op, solution);
    cr.replace(*mm, ins);
    migraphx::run_passes(*mm, {migraphx::dead_code_elimination{}});

    channelwise_gpu_target t;
    p.compile(t);
    migraphx::parameter_map m;
    for(auto&& input : inputs)
        m[input.first] = t.copy_to(input.second);
    m["output"] = t.allocate(out_shape);
    return t.copy_from(p.eval(m).front());
}

// Compile the kernel with `solution` for the given shapes and run it on the gpu
static migraphx::argument run_gpu(const migraphx::shape& x_shape,
                                  const migraphx::shape& w_shape,
                                  const std::vector<std::size_t>& padding,
                                  const migraphx::parameter_map& inputs,
                                  migraphx::value solution)
{
    auto out_shape          = conv_output(x_shape, w_shape, padding);
    solution["num_spatial"] = x_shape.ndim() - 2;
    solution["padding"]     = padding;

    migraphx::gpu::context ctx;
    auto op =
        migraphx::gpu::compile_op("channelwise_conv", ctx, {x_shape, w_shape, out_shape}, solution);

    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x      = mm->add_parameter("x", x_shape);
    auto w      = mm->add_parameter("w", w_shape);
    auto output = mm->add_parameter("output", out_shape);
    mm->add_return({mm->add_instruction(op, x, w, output)});

    channelwise_gpu_target t;
    p.compile(t);
    migraphx::parameter_map m;
    for(auto&& input : inputs)
        m[input.first] = t.copy_to(input.second);
    m["output"] = t.allocate(out_shape);
    return t.copy_from(p.eval(m).front());
}

static bool verify_solution(const std::string& name,
                            const migraphx::shape& x_shape,
                            const migraphx::shape& w_shape,
                            const std::vector<std::size_t>& padding,
                            const migraphx::value& solution)
{
    migraphx::parameter_map inputs;
    inputs["x"] = migraphx::generate_argument(x_shape, 0);
    inputs["w"] = migraphx::generate_argument(w_shape, 1);
    return migraphx::verify_args_with_tolerance(
        name,
        run_gpu(x_shape, w_shape, padding, inputs, solution),
        migraphx::verify::expected{run_ref(x_shape, w_shape, padding, inputs)});
}

// Row runs with tile overhang and padding on a channels-first input
TEST_CASE(channelwise_conv_nrows)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 4, 9, 10}};
    migraphx::shape w_shape{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::value solution{
        {"tile_h", 4}, {"tile_w", 8}, {"noutputs", 1}, {"tile_c", 1}, {"nrows", 4}};
    EXPECT(verify_solution("channelwise_conv_nrows", x_shape, w_shape, {1, 1, 1, 1}, solution));
}

// Row runs with a channel tile on channels-last tensors
TEST_CASE(channelwise_conv_nrows_channels_last)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 3, 9, 10}, perm);
    auto w_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {3, 1, 3, 3}, perm);
    migraphx::value solution{
        {"tile_h", 4}, {"tile_w", 8}, {"noutputs", 2}, {"tile_c", 3}, {"nrows", 4}};
    EXPECT(verify_solution(
        "channelwise_conv_nrows_channels_last", x_shape, w_shape, {1, 1, 1, 1}, solution));
}

// One channels-last input channel feeding several output channels, sharing the halo
TEST_CASE(channelwise_conv_nrows_channel_multiplier)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 1, 9, 10}, perm);
    migraphx::shape w_shape{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::value solution{
        {"tile_h", 2}, {"tile_w", 8}, {"noutputs", 2}, {"tile_c", 4}, {"nrows", 2}};
    EXPECT(verify_solution(
        "channelwise_conv_nrows_channel_multiplier", x_shape, w_shape, {0, 0, 0, 0}, solution));
}

// Channel vectors of half on channels-last tensors with row runs and padding
TEST_CASE(channelwise_conv_channel_vector_half)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::half_type, {2, 16, 9, 10}, perm);
    auto w_shape =
        migraphx::shape::from_permutation(migraphx::shape::half_type, {16, 1, 3, 3}, perm);
    migraphx::value solution{
        {"tile_h", 2}, {"tile_w", 8}, {"noutputs", 2}, {"tile_c", 16}, {"nrows", 2}, {"cvec", 8}};
    EXPECT(verify_solution(
        "channelwise_conv_channel_vector_half", x_shape, w_shape, {1, 1, 1, 1}, solution));
}

// Channel vectors of float with a partial channel tile, so the halo copy stays scalar
TEST_CASE(channelwise_conv_channel_vector_float)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 16, 9, 10}, perm);
    migraphx::shape w_shape{migraphx::shape::float_type, {16, 1, 3, 3}};
    migraphx::value solution{
        {"tile_h", 2}, {"tile_w", 8}, {"noutputs", 1}, {"tile_c", 8}, {"nrows", 1}, {"cvec", 4}};
    EXPECT(verify_solution(
        "channelwise_conv_channel_vector_float", x_shape, w_shape, {0, 0, 0, 0}, solution));
}

// A broadcast input channel cannot be loaded as a vector, so the lanes gather it
TEST_CASE(channelwise_conv_channel_vector_multiplier)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::half_type, {2, 1, 9, 10}, perm);
    migraphx::shape w_shape{migraphx::shape::half_type, {8, 1, 3, 3}};
    migraphx::value solution{
        {"tile_h", 2}, {"tile_w", 8}, {"noutputs", 2}, {"tile_c", 8}, {"nrows", 2}, {"cvec", 4}};
    EXPECT(verify_solution(
        "channelwise_conv_channel_vector_multiplier", x_shape, w_shape, {1, 1, 1, 1}, solution));
}

// ======== bool operands in the fused pointwise ========

// Fused where over a channels-last bool mask; cvec > 1 loads the mask bytes as uint8
// vectors since vec<bool, V> is bit-packed
static bool verify_bool_mask(const std::string& name, std::size_t cvec)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {1, 8, 10, 10}, perm);
    auto w_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {8, 1, 3, 3}, perm);
    auto out_shape = conv_output(x_shape, w_shape, {1, 1, 1, 1});
    migraphx::shape mask_shape{migraphx::shape::bool_type, out_shape.lens(), out_shape.strides()};

    migraphx::parameter_map inputs;
    inputs["p0"] = migraphx::generate_argument(x_shape, 0);
    inputs["p1"] = migraphx::generate_argument(w_shape, 1);
    inputs["p2"] = migraphx::generate_argument(mask_shape, 2);

    migraphx::value solution{
        {"tile_h", 2}, {"tile_w", 8}, {"noutputs", 2}, {"tile_c", 8}, {"nrows", 1}, {"cvec", cvec}};
    auto gpu_result = run_gpu_fused(
        {x_shape, w_shape, mask_shape},
        out_shape,
        {1, 1, 1, 1},
        inputs,
        solution,
        [&](migraphx::module& pm, const std::vector<migraphx::instruction_ref>& params) {
            auto zero = pm.add_literal(migraphx::literal{migraphx::shape{x_shape.type()}, {0.0f}});
            return pm.add_instruction(migraphx::make_op("where"), params[1], params[0], zero);
        });

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("p0", x_shape);
    auto w    = mm->add_parameter("p1", w_shape);
    auto mask = mm->add_parameter("p2", mask_shape);
    auto conv = add_ref_conv(*mm, x, w, {1, 1, 1, 1});
    auto zero = mm->add_literal(migraphx::literal{migraphx::shape{x_shape.type()}, {0.0f}});
    auto zb   = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", out_shape.lens()}}), zero);
    mm->add_instruction(migraphx::make_op("where"), mask, conv, zb);
    p.compile(migraphx::make_target("ref"));
    auto ref_result = p.eval(inputs).front();

    return migraphx::verify_args_with_tolerance(
        name, gpu_result, migraphx::verify::expected{ref_result});
}

TEST_CASE(channelwise_conv_bool_mask)
{
    EXPECT(verify_bool_mask("channelwise_conv_bool_mask", 1));
}

TEST_CASE(channelwise_conv_bool_mask_vector)
{
    EXPECT(verify_bool_mask("channelwise_conv_bool_mask_vector", 4));
}

// Fused comparison producing a bool output; the conv result must reach the pointwise op
// at conv precision, not pre-cast to the bool output type
static bool verify_bool_output(const std::string& name, std::size_t cvec)
{
    std::vector<int64_t> perm = {0, 2, 3, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {1, 8, 10, 10}, perm);
    auto w_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {8, 1, 3, 3}, perm);
    auto outf = conv_output(x_shape, w_shape, {1, 1, 1, 1});
    migraphx::shape out_shape{migraphx::shape::bool_type, outf.lens(), outf.strides()};

    migraphx::parameter_map inputs;
    inputs["p0"] = migraphx::generate_argument(x_shape, 0);
    inputs["p1"] = migraphx::generate_argument(w_shape, 1);

    migraphx::value solution{
        {"tile_h", 2}, {"tile_w", 8}, {"noutputs", 2}, {"tile_c", 8}, {"nrows", 1}, {"cvec", cvec}};
    auto to_bool = migraphx::make_op(
        "convert", {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}});
    auto gpu_result = run_gpu_fused(
        {x_shape, w_shape},
        out_shape,
        {1, 1, 1, 1},
        inputs,
        solution,
        [&](migraphx::module& pm, const std::vector<migraphx::instruction_ref>& params) {
            auto half = pm.add_literal(migraphx::literal{migraphx::shape{x_shape.type()}, {0.5f}});
            auto g    = pm.add_instruction(migraphx::make_op("greater"), params[0], half);
            return pm.add_instruction(to_bool, g);
        });

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("p0", x_shape);
    auto w    = mm->add_parameter("p1", w_shape);
    auto conv = add_ref_conv(*mm, x, w, {1, 1, 1, 1});
    auto half = mm->add_literal(migraphx::literal{migraphx::shape{x_shape.type()}, {0.5f}});
    auto hb   = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", out_shape.lens()}}), half);
    auto g = mm->add_instruction(migraphx::make_op("greater"), conv, hb);
    mm->add_instruction(to_bool, g);
    p.compile(migraphx::make_target("ref"));
    auto ref_result = p.eval(inputs).front();

    return migraphx::verify_args_with_tolerance(
        name, gpu_result, migraphx::verify::expected{ref_result});
}

TEST_CASE(channelwise_conv_bool_output)
{
    EXPECT(verify_bool_output("channelwise_conv_bool_output", 1));
}

TEST_CASE(channelwise_conv_bool_output_vector)
{
    EXPECT(verify_bool_output("channelwise_conv_bool_output_vector", 4));
}

// 1-D convolution on a channels-first input; tile_h defaults to 1
TEST_CASE(channelwise_conv_1d)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 8, 100}};
    migraphx::shape w_shape{migraphx::shape::float_type, {8, 1, 3}};
    migraphx::value solution{{"tile_w", 64}, {"noutputs", 2}, {"tile_c", 1}, {"nrows", 1}};
    EXPECT(verify_solution("channelwise_conv_1d", x_shape, w_shape, {1, 1}, solution));
}

// 1-D convolution on channels-last tensors with a channel tile and vector
TEST_CASE(channelwise_conv_1d_channels_last)
{
    std::vector<int64_t> perm = {0, 2, 1};
    auto x_shape =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 8, 100}, perm);
    auto w_shape = migraphx::shape::from_permutation(migraphx::shape::float_type, {8, 1, 3}, perm);
    migraphx::value solution{
        {"tile_w", 16}, {"noutputs", 2}, {"tile_c", 8}, {"nrows", 1}, {"cvec", 4}};
    EXPECT(verify_solution(
        "channelwise_conv_1d_channels_last", x_shape, w_shape, {1, 1}, solution));
}

// tile_h has no meaning for 1-D convolutions, so compiling with it set must fail
TEST_CASE(channelwise_conv_1d_tile_h_throws)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 8, 100}};
    migraphx::shape w_shape{migraphx::shape::float_type, {8, 1, 3}};
    auto out_shape = conv_output(x_shape, w_shape, {1, 1});
    migraphx::value solution{{"tile_h", 8},
                             {"tile_w", 16},
                             {"noutputs", 1},
                             {"tile_c", 1},
                             {"nrows", 1},
                             {"num_spatial", 1},
                             {"padding", {1, 1}}};
    migraphx::gpu::context ctx;
    EXPECT(test::throws([&] {
        migraphx::gpu::compile_op("channelwise_conv", ctx, {x_shape, w_shape, out_shape}, solution);
    }));
}

static migraphx::instruction_ref add_channelwise_conv_1d(migraphx::program& p,
                                                         const migraphx::operation& op)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {1, 8, 64}};
    migraphx::shape w_shape{migraphx::shape::float_type, {8, 1, 3}};
    auto out_shape = conv_output(x_shape, w_shape, {1, 1});
    auto* mm       = p.get_main_module();
    auto x         = mm->add_parameter("x", x_shape);
    auto w         = mm->add_parameter("w", w_shape);
    auto alloc     = mm->add_parameter("output", out_shape);
    return mm->add_instruction(make_precompile_op(op), x, w, alloc);
}

// Each distinct solution appears once
static bool unique_solutions(const std::vector<migraphx::value>& solutions)
{
    return std::all_of(solutions.begin(), solutions.end(), [&](const migraphx::value& s) {
        return std::count(solutions.begin(), solutions.end(), s) == 1;
    });
}

// Tuning for 1-D only proposes tile_h=1 and sizes blocks by the threads it launches
TEST_CASE(channelwise_conv_1d_tuning)
{
    migraphx::program p;
    auto op = migraphx::make_op("gpu::channelwise_conv",
                                {{"num_spatial", 1}, {"padding", {1, 1}}});
    auto ins = add_channelwise_conv_1d(p, op);
    migraphx::gpu::context ctx;
    auto wave = ctx.get_current_device().get_wavefront_size();
    for(bool exhaustive : {false, true})
    {
        auto tc = migraphx::gpu::get_tuning_config(ctx, ins, op, exhaustive);
        EXPECT(tc.has_value());
        EXPECT(not tc->solutions.empty());
        EXPECT(std::all_of(
            tc->solutions.begin(), tc->solutions.end(), [&](const migraphx::value& s) {
                auto threads = s.at("tile_w").to<std::size_t>() *
                               (s.at("tile_c").to<std::size_t>() / s.at("cvec").to<std::size_t>());
                return s.at("tile_h").to<std::size_t>() == 1 and
                       s.at("nrows").to<std::size_t>() == 1 and threads >= wave and
                       threads % wave == 0;
            }));
        EXPECT(unique_solutions(tc->solutions));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
