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
 *
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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <numeric>
#include <test.hpp>

// Compiles the direct conv kernel with a chosen tuning solution and layout and checks the
// result against the reference target. Going through the target's passes would let the
// heuristic decide whether the kernel runs and the tuner pick the solution.

struct direct_conv_gpu_target : migraphx::gpu::target
{
    std::string name() const { return "direct_conv"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {migraphx::gpu::write_literals{}};
    }
};

// Conv attributes shared by the reference conv and the kernel; an omitted stride or
// dilation means ones
struct conv_attrs
{
    std::vector<std::size_t> padding;
    std::vector<std::size_t> stride;
    std::vector<std::size_t> dilation;

    conv_attrs(std::vector<std::size_t> p,
               const std::vector<std::size_t>& s = {},
               const std::vector<std::size_t>& d = {})
        : padding(std::move(p)), stride(or_ones(s)), dilation(or_ones(d))
    {
    }

    migraphx::value to_value() const
    {
        return {{"padding", padding}, {"stride", stride}, {"dilation", dilation}};
    }

    migraphx::operation conv_op() const { return migraphx::make_op("convolution", to_value()); }

    private:
    std::vector<std::size_t> or_ones(const std::vector<std::size_t>& attr) const
    {
        return attr.empty() ? std::vector<std::size_t>(padding.size() / 2, 1) : attr;
    }
};

static migraphx::shape channels_last(migraphx::shape::type_t t,
                                     const std::vector<std::size_t>& lens)
{
    std::vector<int64_t> perm(lens.size());
    std::iota(perm.begin() + 1, perm.end() - 1, 2);
    perm.back() = 1;
    return migraphx::shape::from_permutation(t, lens, perm);
}

static migraphx::argument run_ref(const migraphx::shape& x_shape,
                                  const migraphx::shape& w_shape,
                                  const conv_attrs& attrs,
                                  const migraphx::parameter_map& inputs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", x_shape);
    auto w   = mm->add_parameter("w", w_shape);
    mm->add_instruction(attrs.conv_op(), x, w);
    p.compile(migraphx::make_target("ref"));
    return p.eval(inputs).front();
}

static migraphx::argument run_program(migraphx::program p,
                                      const migraphx::shape& out_shape,
                                      const migraphx::parameter_map& inputs)
{
    direct_conv_gpu_target t;
    p.compile(t);
    migraphx::parameter_map m;
    for(const auto& [name, arg] : inputs)
        m[name] = t.copy_to(arg);
    m["output"] = t.allocate(out_shape);
    return t.copy_from(p.eval(m).front());
}

// Compile the kernel with `solution` (the default one when empty) and run it on the gpu
static migraphx::argument run_gpu(const migraphx::shape& x_shape,
                                  const migraphx::shape& w_shape,
                                  const conv_attrs& attrs,
                                  const migraphx::parameter_map& inputs,
                                  migraphx::value solution)
{
    auto out_shape = attrs.conv_op().compute_shape({x_shape, w_shape});
    for(const auto& attr : attrs.to_value())
        solution.insert(attr);

    migraphx::gpu::context ctx;
    auto op =
        migraphx::gpu::compile_op("direct_conv", ctx, {x_shape, w_shape, out_shape}, solution);

    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x      = mm->add_parameter("x", x_shape);
    auto w      = mm->add_parameter("w", w_shape);
    auto output = mm->add_parameter("output", out_shape);
    mm->add_return({mm->add_instruction(op, x, w, output)});
    return run_program(std::move(p), out_shape, inputs);
}

static bool verify_solution(const std::string& name,
                            const migraphx::shape& x_shape,
                            const migraphx::shape& w_shape,
                            const conv_attrs& attrs,
                            const migraphx::value& solution)
{
    migraphx::parameter_map inputs;
    inputs["x"] = migraphx::generate_argument(x_shape, 0);
    inputs["w"] = migraphx::generate_argument(w_shape, 1);
    return migraphx::verify_args_with_tolerance(
        name,
        run_gpu(x_shape, w_shape, attrs, inputs, solution),
        migraphx::verify::expected{run_ref(x_shape, w_shape, attrs, inputs)});
}

static migraphx::value config(std::size_t tile_h,
                              std::size_t tile_w,
                              std::size_t noutputs,
                              std::size_t nrows,
                              std::size_t tile_k,
                              std::size_t kvec,
                              std::size_t tile_c)
{
    return {{"tile_h", tile_h},
            {"tile_w", tile_w},
            {"noutputs", noutputs},
            {"nrows", nrows},
            {"tile_k", tile_k},
            {"kvec", kvec},
            {"tile_c", tile_c}};
}

// Several input channel stages, row runs and output channel vectors with padding and a
// partial tile on channels-first tensors
TEST_CASE(direct_conv_channels_first)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 6, 9, 10}};
    migraphx::shape w_shape{migraphx::shape::float_type, {8, 6, 3, 3}};
    EXPECT(verify_solution("direct_conv_channels_first",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}},
                           config(4, 8, 2, 2, 8, 4, 2)));
}

// Channels-last float tensors: the halo copies channel vectors and the lanes run over the
// output channels first
TEST_CASE(direct_conv_channels_last)
{
    auto x_shape = channels_last(migraphx::shape::float_type, {2, 8, 9, 10});
    auto w_shape = channels_last(migraphx::shape::float_type, {16, 8, 3, 3});
    EXPECT(verify_solution("direct_conv_channels_last",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}},
                           config(2, 8, 2, 2, 16, 2, 8)));
}

// Channels-last half tensors staging 4 of the 16 input channels at a time, with wide output
// channel vectors
TEST_CASE(direct_conv_channels_last_half)
{
    auto x_shape = channels_last(migraphx::shape::half_type, {1, 16, 11, 12});
    auto w_shape = channels_last(migraphx::shape::half_type, {16, 16, 3, 3});
    EXPECT(verify_solution("direct_conv_channels_last_half",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}},
                           config(2, 16, 1, 2, 16, 8, 4)));
}

// Three output channels per lane are not a power of two, so their weights load one at a time
TEST_CASE(direct_conv_three_outputs)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {1, 12, 13, 17}};
    migraphx::shape w_shape{migraphx::shape::float_type, {3, 12, 5, 5}};
    EXPECT(verify_solution("direct_conv_three_outputs",
                           x_shape,
                           w_shape,
                           {{2, 2, 2, 2}},
                           config(4, 8, 2, 4, 3, 3, 4)));
}

// Half input channel pairs for three output channels per lane, which load their weight
// pairs one at a time
TEST_CASE(direct_conv_three_outputs_half)
{
    migraphx::shape x_shape{migraphx::shape::half_type, {1, 12, 13, 17}};
    migraphx::shape w_shape{migraphx::shape::half_type, {3, 12, 5, 5}};
    EXPECT(verify_solution("direct_conv_three_outputs_half",
                           x_shape,
                           w_shape,
                           {{2, 2, 2, 2}},
                           config(4, 8, 2, 4, 3, 3, 4)));
}

// An odd half input channel tile has no channel pairs and reduces one channel at a time
TEST_CASE(direct_conv_odd_channels_half)
{
    migraphx::shape x_shape{migraphx::shape::half_type, {1, 3, 17, 18}};
    migraphx::shape w_shape{migraphx::shape::half_type, {8, 3, 4, 4}};
    EXPECT(verify_solution("direct_conv_odd_channels_half",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}, {2, 2}},
                           config(2, 16, 1, 2, 8, 2, 3)));
}

// A 1x1 convolution reads no halo around the tile
TEST_CASE(direct_conv_1x1)
{
    auto x_shape = channels_last(migraphx::shape::half_type, {2, 16, 7, 9});
    auto w_shape = channels_last(migraphx::shape::half_type, {4, 16, 1, 1});
    EXPECT(verify_solution(
        "direct_conv_1x1", x_shape, w_shape, {{0, 0, 0, 0}}, config(4, 8, 2, 2, 4, 4, 8)));
}

// Stride 2 with a 4x4 filter, as in encoder downsampling
TEST_CASE(direct_conv_stride)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {1, 3, 17, 18}};
    migraphx::shape w_shape{migraphx::shape::float_type, {8, 3, 4, 4}};
    EXPECT(verify_solution("direct_conv_stride",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}, {2, 2}},
                           config(2, 16, 1, 2, 8, 2, 3)));
}

// Dilation 2: each output row of a row run reads every other halo row
TEST_CASE(direct_conv_dilation)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {1, 4, 12, 12}};
    migraphx::shape w_shape{migraphx::shape::float_type, {4, 4, 3, 3}};
    EXPECT(verify_solution("direct_conv_dilation",
                           x_shape,
                           w_shape,
                           {{2, 2, 2, 2}, {1, 1}, {2, 2}},
                           config(4, 8, 1, 2, 4, 4, 2)));
}

// Asymmetric padding with a stride that skips the last end-padding row
TEST_CASE(direct_conv_asymmetric_padding)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {1, 2, 10, 11}};
    migraphx::shape w_shape{migraphx::shape::float_type, {2, 2, 3, 2}};
    EXPECT(verify_solution("direct_conv_asymmetric_padding",
                           x_shape,
                           w_shape,
                           {{0, 1, 2, 0}, {2, 2}},
                           config(4, 8, 1, 1, 2, 1, 1)));
}

// 1-D convolution with stride 2
TEST_CASE(direct_conv_1d)
{
    migraphx::shape x_shape{migraphx::shape::float_type, {2, 8, 101}};
    migraphx::shape w_shape{migraphx::shape::float_type, {4, 8, 5}};
    EXPECT(verify_solution(
        "direct_conv_1d", x_shape, w_shape, {{2, 2}, {2}}, config(1, 32, 2, 1, 4, 2, 4)));
}

// The last block's channel tile runs past the 12 output channels; the lanes of its missing
// channel vector skip their stores
TEST_CASE(direct_conv_partial_channel_tile)
{
    auto x_shape = channels_last(migraphx::shape::float_type, {1, 6, 9, 10});
    auto w_shape = channels_last(migraphx::shape::float_type, {12, 6, 3, 3});
    EXPECT(verify_solution("direct_conv_partial_channel_tile",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}},
                           config(2, 8, 2, 2, 8, 4, 3)));
}

// Ten output channels split a lane's channel vector, so its stores are masked one by one
TEST_CASE(direct_conv_partial_channel_vector)
{
    migraphx::shape x_shape{migraphx::shape::half_type, {2, 4, 9, 10}};
    migraphx::shape w_shape{migraphx::shape::half_type, {10, 4, 3, 3}};
    EXPECT(verify_solution("direct_conv_partial_channel_vector",
                           x_shape,
                           w_shape,
                           {{1, 1, 1, 1}},
                           config(4, 8, 1, 2, 8, 4, 2)));
}

// Without a solution the kernel compiles with the first default candidate
TEST_CASE(direct_conv_default_solution)
{
    migraphx::shape x_shape{migraphx::shape::half_type, {1, 48, 20, 21}};
    migraphx::shape w_shape{migraphx::shape::half_type, {47, 48, 3, 3}};
    EXPECT(verify_solution("direct_conv_default_solution", x_shape, w_shape, {{1, 1, 1, 1}}, {}));
}

// A fused add + relu reads a broadcast bias at each output, mirroring
// find_tiled_conv_pointwise
TEST_CASE(direct_conv_fused_bias_relu)
{
    auto x_shape = channels_last(migraphx::shape::float_type, {1, 4, 9, 10});
    auto w_shape = channels_last(migraphx::shape::float_type, {8, 4, 3, 3});
    conv_attrs attrs{{1, 1, 1, 1}};
    auto out_shape = attrs.conv_op().compute_shape({x_shape, w_shape});
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};

    migraphx::parameter_map inputs;
    inputs["x"] = migraphx::generate_argument(x_shape, 0);
    inputs["w"] = migraphx::generate_argument(w_shape, 1);
    inputs["b"] = migraphx::generate_argument(b_shape, 2);

    migraphx::program ref;
    {
        auto* mm  = ref.get_main_module();
        auto x    = mm->add_parameter("x", x_shape);
        auto w    = mm->add_parameter("w", w_shape);
        auto b    = mm->add_parameter("b", b_shape);
        auto conv = mm->add_instruction(attrs.conv_op(), x, w);
        auto bb   = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", out_shape.lens()}}), b);
        auto add = mm->add_instruction(migraphx::make_op("add"), conv, bb);
        mm->add_instruction(migraphx::make_op("relu"), add);
    }
    ref.compile(migraphx::make_target("ref"));

    migraphx::program p;
    {
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", x_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto b   = mm->add_parameter("b", b_shape);
        auto bb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", out_shape.lens()}}), b);
        auto output = mm->add_parameter("output", out_shape);

        auto* pm = p.create_module("post");
        pm->set_bypass();
        auto c   = pm->add_parameter("x0", migraphx::shape{migraphx::shape::float_type});
        auto pb  = pm->add_parameter("x1", migraphx::shape{migraphx::shape::float_type});
        auto add = pm->add_instruction(migraphx::make_op("add"), c, pb);
        pm->add_return({pm->add_instruction(migraphx::make_op("relu"), add)});

        auto conv_op = migraphx::make_op("gpu::direct_conv", attrs.to_value());
        auto pre     = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(conv_op)}, {"output_shape", migraphx::to_value(out_shape)}});
        auto ins = mm->add_instruction(pre, {x, w, bb, output}, {pm});
        mm->add_return({ins});

        migraphx::gpu::context ctx;
        auto cr = migraphx::gpu::compile(ctx, ins, conv_op, config(2, 8, 2, 2, 8, 4, 4));
        cr.replace(*mm, ins);
        migraphx::run_passes(*mm, {migraphx::dead_code_elimination{}});
    }

    EXPECT(
        migraphx::verify_args_with_tolerance("direct_conv_fused_bias_relu",
                                             run_program(std::move(p), out_shape, inputs),
                                             migraphx::verify::expected{ref.eval(inputs).front()}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
