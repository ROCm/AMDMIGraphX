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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <group.hpp>
#include <test.hpp>
#include <cstdlib>

static migraphx::gpu::context& get_context()
{
    static migraphx::gpu::context ctx;
    return ctx;
}

static void skip_if_not_gfx1201()
{
    const auto& gfx = get_context().get_current_device().get_gfx_name();
    if(not migraphx::starts_with(gfx, "gfx1201"))
        test::skip("test requires gfx1201, got: " + gfx);
}

#ifdef MIGRAPHX_HAS_MLSS_HEADERS

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_mlss{&get_context()}, migraphx::dead_code_elimination{}});
}

// Set MIGRAPHX_MLSS_USE_SPECIFIC_OPS=mha,conv at static-init time, before any test runs.
// This must happen before the first call to string_value_of(), which caches its result.
const int mlss_env_init = ([] {
#ifdef _WIN32
    _putenv_s("MIGRAPHX_MLSS_USE_SPECIFIC_OPS", "mha,conv");
#else
    setenv("MIGRAPHX_MLSS_USE_SPECIFIC_OPS", "mha,conv", /*overwrite=*/1); // NOLINT(cert-env33-c)
#endif
}(), 0);


// Build the pre-pass "group" attention program with shape {1,8,4096,40}.
// The submodule contains a @literal scale (half) and a minimal attention body.
static migraphx::program make_attention_program(const migraphx::shape& qkv_shape, float scale_val)
{
    const migraphx::shape s_scale{migraphx::shape::half_type, {1}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("query", qkv_shape);
    auto k   = mm->add_parameter("key",   qkv_shape);
    auto v   = mm->add_parameter("value", qkv_shape);

    auto grp = add_group(
        p,
        "attn0",
        "attention",
        {q, k, v},
        {"x0", "x1", "x2"},
        [=](auto* gm, const auto& inputs) {
            // Scale literal that fuse_mlss reads from the submodule
            gm->add_literal(migraphx::literal{s_scale, {scale_val}});

            auto kt = gm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
            auto scores = gm->add_instruction(migraphx::make_op("dot"), inputs[0], kt);
            auto out    = gm->add_instruction(migraphx::make_op("dot"), scores, inputs[2]);
            return std::vector<migraphx::instruction_ref>{out};
        });

    mm->add_return({grp});
    return p;
}

// Verify that fuse_mlss replaces the "group" attention instruction with "mlss_mha"
// for the supported shape {1, 8, 4096, 40}, and that the resulting mlss_mha has
// 4 inputs (Q, K, V, scale_literal) and the correct output shape.
TEST_CASE(mlss_mha_attention_1x8x4096x40)
{
    skip_if_not_gfx1201();
    const migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 8, 4096, 40}};
    const float scale_val = 1.0f / std::sqrt(40.0f);

    migraphx::program p = make_attention_program(qkv_shape, scale_val);
    run_pass(p);

    auto* mm = p.get_main_module();

    bool found_mlss_mha = false;
    bool found_group    = false;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        auto n = ins->name();
        if(n == "gpu::mlss_mha")
            found_mlss_mha = true;
        if(n == "group")
            found_group = true;
    }

    // The group must be replaced by mlss_mha
    EXPECT(found_mlss_mha);
    EXPECT(not found_group);

    // Validate inputs and output shape of mlss_mha
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() != "mlss_mha")
            continue;

        // Inputs: Q, K, V, scale_literal
        EXPECT(ins->inputs().size() == 4);

        // 4th input must be a half scalar literal (the scale)
        auto scale_in = ins->inputs()[3];
        EXPECT(scale_in->name() == "@literal");
        EXPECT(scale_in->get_shape().type() == migraphx::shape::half_type);

        // Output shape must match Q/K/V shape
        EXPECT(ins->get_shape() == qkv_shape);
    }
}

// Verify that fuse_mlss does NOT fuse a "group" whose query shape differs from
// the guarded shape {1, 8, 4096, 40}.
TEST_CASE(mlss_mha_attention_wrong_shape_not_fused)
{
    skip_if_not_gfx1201();
    // {1, 8, 512, 64} is not the supported shape
    const migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 8, 512, 64}};
    const float scale_val = 1.0f / std::sqrt(64.0f);

    migraphx::program p = make_attention_program(qkv_shape, scale_val);
    run_pass(p);

    auto* mm = p.get_main_module();

    bool found_mlss_mha = false;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() == "mlss_mha")
            found_mlss_mha = true;
    }

    EXPECT(not found_mlss_mha);
}


// Build the pre-pass program for conv+bias+relu:
//   relu(add(convolution(data, weight_literal), broadcast(bias_literal)))
// The shapes match the VGG-19 first-layer entry in conv_mxn_shapes:
//   act  {1, 3, 224, 224}, weight {64, 3, 3, 3}, out {1, 64, 224, 224}, pad {1,1,1,1}, stride {1,1}
static migraphx::program make_conv_bias_relu_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    const migraphx::shape act_shape{migraphx::shape::float_type, {1, 3, 224, 224}};
    const migraphx::shape wt_shape{migraphx::shape::float_type, {64, 3, 3, 3}};
    const migraphx::shape bias_shape{migraphx::shape::float_type, {64}};
    const migraphx::shape out_shape{migraphx::shape::float_type, {1, 64, 224, 224}};

    auto data   = mm->add_parameter("data_0", act_shape);
    auto weight = mm->add_literal(migraphx::literal{wt_shape, std::vector<float>(wt_shape.elements(), 0.0f)});
    auto bias   = mm->add_literal(migraphx::literal{bias_shape, std::vector<float>(bias_shape.elements(), 0.0f)});

    auto conv = mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1, 1, 1}},
                           {"stride",  {1, 1}},
                           {"dilation", {1, 1}},
                           {"group", 1},
                           {"padding_mode", 0}}),
        data, weight);

    auto bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 64, 224, 224}}}), bias);

    auto add  = mm->add_instruction(migraphx::make_op("add"),  conv, bcast);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);

    mm->add_return({relu});
    return p;
}

// Verify that fuse_mlss fuses conv+bias+relu into a single gpu::mlss_conv instruction
// with has_bias=true and activation_mode=relu (uint8 value 4).
TEST_CASE(mlss_conv_bias_relu_vgg19_first_layer)
{
    skip_if_not_gfx1201();

    migraphx::program p = make_conv_bias_relu_program();
    run_pass(p);

    auto* mm = p.get_main_module();

    bool found_mlss_conv = false;
    bool found_conv      = false;
    bool found_relu      = false;
    bool found_add       = false;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        auto n = ins->name();
        if(n == "gpu::mlss_conv")  found_mlss_conv = true;
        if(n == "convolution")     found_conv      = true;
        if(n == "relu")            found_relu      = true;
        if(n == "add")             found_add       = true;
    }

    // conv+add+relu must be replaced by a single mlss_conv
    EXPECT(found_mlss_conv);
    EXPECT(not found_conv);
    EXPECT(not found_relu);
    EXPECT(not found_add);

    // Validate the fused instruction
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() != "gpu::mlss_conv")
            continue;

        // args: [input, weight, bias, output_alloc]
        EXPECT(ins->inputs().size() == 4);

        // Output shape must match the conv output
        const migraphx::shape expected_out{migraphx::shape::float_type, {1, 64, 224, 224}};
        EXPECT(ins->get_shape() == expected_out);

        // Check has_bias and activation_mode via reflected value
        auto val = ins->get_operator().to_value();
        EXPECT(val.at("has_bias").to<bool>());
        EXPECT(val.at("activation_mode").to<uint8_t>() == 4); // relu
    }
}

// Verify that an unsupported conv shape (not in conv_mxn_shapes) is NOT fused.
TEST_CASE(mlss_conv_bias_relu_unsupported_shape_not_fused)
{
    skip_if_not_gfx1201();

    migraphx::program p;
    auto* mm = p.get_main_module();

    // Use a shape that is NOT in conv_mxn_shapes: {1, 32, 112, 112} output
    const migraphx::shape act_shape{migraphx::shape::float_type, {1, 3, 112, 112}};
    const migraphx::shape wt_shape{migraphx::shape::float_type, {32, 3, 3, 3}};
    const migraphx::shape bias_shape{migraphx::shape::float_type, {32}};

    auto data   = mm->add_parameter("data_0", act_shape);
    auto weight = mm->add_literal(migraphx::literal{wt_shape, std::vector<float>(wt_shape.elements(), 0.0f)});
    auto bias   = mm->add_literal(migraphx::literal{bias_shape, std::vector<float>(bias_shape.elements(), 0.0f)});

    auto conv = mm->add_instruction(
        migraphx::make_op("convolution",
                          {{"padding", {1, 1, 1, 1}},
                           {"stride",  {1, 1}},
                           {"dilation", {1, 1}},
                           {"group", 1},
                           {"padding_mode", 0}}),
        data, weight);

    auto bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 32, 112, 112}}}), bias);

    auto add  = mm->add_instruction(migraphx::make_op("add"),  conv, bcast);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
    mm->add_return({relu});

    run_pass(p);

    bool found_mlss_conv = false;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() == "gpu::mlss_conv")
            found_mlss_conv = true;
    }

    EXPECT(not found_mlss_conv);
}
#endif // MIGRAPHX_HAS_MLSS_HEADERS

int main(int argc, const char* argv[]) { test::run(argc, argv); }
