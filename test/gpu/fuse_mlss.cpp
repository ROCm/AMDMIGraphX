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

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_mlss{&get_context()}, migraphx::dead_code_elimination{}});
}

// Set MIGRAPHX_MLSS_USE_SPECIFIC_OPS=mha at static-init time, before any test runs.
// This must happen before the first call to string_value_of(), which caches its result.
const int mlss_env_init = ([] {
#ifdef _WIN32
    _putenv_s("MIGRAPHX_MLSS_USE_SPECIFIC_OPS", "mha");
#else
    setenv("MIGRAPHX_MLSS_USE_SPECIFIC_OPS", "mha", /*overwrite=*/1); // NOLINT(cert-env33-c)
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
