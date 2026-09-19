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
#include <migraphx/bit_cast.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <test.hpp>

#ifdef MIGRAPHX_USE_AMDMLSS

static migraphx::gpu::context& get_context()
{
    static migraphx::gpu::context ctx;
    return ctx;
}

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p,
        {migraphx::gpu::fuse_mlss{.ctx = &get_context(), .use_specific_ops = {"conv"}},
         migraphx::dead_code_elimination{}});
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

    auto data = mm->add_parameter("data_0", act_shape);
    auto weight =
        mm->add_literal(migraphx::literal{wt_shape, std::vector<float>(wt_shape.elements(), 0.0f)});
    auto bias = mm->add_literal(
        migraphx::literal{bias_shape, std::vector<float>(bias_shape.elements(), 0.0f)});

    auto conv = mm->add_instruction(migraphx::make_op("convolution",
                                                      {{"padding", {1, 1, 1, 1}},
                                                       {"stride", {1, 1}},
                                                       {"dilation", {1, 1}},
                                                       {"group", 1},
                                                       {"padding_mode", 0}}),
                                    data,
                                    weight);

    auto bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 64, 224, 224}}}), bias);

    auto add  = mm->add_instruction(migraphx::make_op("add"), conv, bcast);
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);

    mm->add_return({relu});
    return p;
}

// Verify that fuse_mlss fuses conv+bias+relu into a single gpu::mlss_conv instruction
// with has_bias=true and activation_mode=relu (uint8 value 4).
TEST_CASE(mlss_conv_bias_relu_vgg19_first_layer)
{

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
        if(n == "gpu::mlss_conv")
            found_mlss_conv = true;
        if(n == "convolution")
            found_conv = true;
        if(n == "relu")
            found_relu = true;
        if(n == "add")
            found_add = true;
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

        // args: [input, weight, bias]
        EXPECT(ins->inputs().size() == 3);

        // Output shape must match the conv output
        const migraphx::shape expected_out{migraphx::shape::float_type, {1, 64, 224, 224}};
        EXPECT(ins->get_shape() == expected_out);

        // Check has_bias and activation_mode via reflected value
        auto val = ins->get_operator().to_value();
        EXPECT(val.at("has_bias").to<bool>());
        EXPECT(val.at("activation_mode").to<uint8_t>() == 4); // relu
    }
}

// Build a bare conv program with the given batch size. The remaining shapes
// match the instruction that stalled on Navi48: C=K=1280, 8x8, 3x3, pad 1,
// stride 1, fp16.
static migraphx::program make_conv_program(std::size_t batch)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    const migraphx::shape act_shape{migraphx::shape::half_type, {batch, 1280, 8, 8}};
    const migraphx::shape wt_shape{migraphx::shape::half_type, {1280, 1280, 3, 3}};

    auto data   = mm->add_parameter("data_0", act_shape);
    auto weight = mm->add_literal(migraphx::generate_literal(wt_shape));

    auto conv = mm->add_instruction(migraphx::make_op("convolution",
                                                      {{"padding", {1, 1, 1, 1}},
                                                       {"stride", {1, 1}},
                                                       {"dilation", {1, 1}},
                                                       {"group", 1},
                                                       {"padding_mode", 0}}),
                                    data,
                                    weight);

    mm->add_return({conv});
    return p;
}

// The nGroups the shader is told about in kernel_args[5] must equal the number
// of workgroups the dispatch actually launches (per conv group). The two are
// produced at different call sites -- n_groups in query_mlss_conv_binary(), the
// grid in the jit compiler -- so nothing but this check keeps them in step. A
// batch factor that appeared in both used to cancel in the grid while shrinking
// the argument, leaving every workgroup past the first nGroups to stride its
// tile loop off the end of its assignment and write out of bounds.
static void check_ngroups_matches_dispatch(std::size_t batch)
{
    migraphx::program p = make_conv_program(batch);
    run_pass(p);

    auto* mm                           = p.get_main_module();
    migraphx::instruction_ref mlss_ins = mm->end();
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() == "gpu::mlss_conv")
            mlss_ins = ins;
    }
    // AMDMLSS has no kernel for this configuration on this device, so there is
    // no dispatch to check.
    if(mlss_ins == mm->end())
        return;

    // compile_op() expects [input, weight, output_buffer]; the output buffer is
    // only appended once lowering runs, which is after this pass.
    const std::vector<migraphx::shape> inputs = {mlss_ins->inputs()[0]->get_shape(),
                                                 mlss_ins->inputs()[1]->get_shape(),
                                                 mlss_ins->get_shape()};
    auto op                                   = migraphx::gpu::compile_op(
        "gpu::mlss_conv", get_context(), inputs, mlss_ins->get_operator().to_value());
    auto cop = migraphx::any_cast<migraphx::gpu::code_object_op>(op);

    EXPECT(cop.local > 0);
    EXPECT(cop.global % cop.local == 0);
    const std::size_t workgroups = cop.global / cop.local;

    // Unpack the argument the way kernel_argument_value packed it.
    const auto& ng_arg = cop.kernel_args.at(5);
    EXPECT(ng_arg.data.size() == sizeof(std::int32_t));
    std::array<char, sizeof(std::int32_t)> ng_bytes{};
    std::copy(ng_arg.data.begin(), ng_arg.data.end(), ng_bytes.begin());
    auto ng = migraphx::bit_cast<std::int32_t>(ng_bytes);

    EXPECT(ng > 0);
    // group == 1 for this conv, so the workgroup count is nGroups exactly.
    EXPECT(static_cast<std::size_t>(ng) == workgroups);
}

// Batch must not change the relationship between the two. Before the fix batch
// 1 held (the stray factor was 1) while batch 2 launched twice the workgroups
// the kernel was told about.
TEST_CASE(mlss_conv_ngroups_matches_dispatch)
{
    for(std::size_t batch : std::vector<std::size_t>{1, 2, 4})
        check_ngroups_matches_dispatch(batch);
}

#endif // MIGRAPHX_USE_AMDMLSS

int main(int argc, const char* argv[]) { test::run(argc, argv); }
