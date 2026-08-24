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

#include <vector>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <test.hpp>

// splitKFactor=1 and splitKFactor=4 perf configs in the legacy positional format.
static constexpr const char* split_k1_perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0";
static constexpr const char* split_k4_perf_config = "gemm:v1:64,64,64,1,1,4,16,4,2,0,0";

static bool mlir_enabled()
{
    migraphx::module m;
    auto x   = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto w   = m.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto dot = m.add_instruction(migraphx::make_op("dot"), x, w);
    m.add_return({dot});
    return not migraphx::gpu::dump_mlir(m).empty();
}

static migraphx::instruction_ref
add_fused_mlir_op(migraphx::program& p,
                  migraphx::module& fused_mod,
                  const std::vector<migraphx::instruction_ref>& inputs)
{
    auto* mm = p.get_main_module();
    return mm->add_instruction(
        migraphx::make_op("gpu::mlir_op", {{"op", migraphx::to_value(migraphx::make_op("dot"))}}),
        inputs,
        {&fused_mod});
}

static migraphx::instruction_ref build_dot_add_tanh_mlir(migraphx::program& p,
                                                         bool dot_has_multiple_outputs = false)
{
    migraphx::shape s_a{migraphx::shape::float_type, {1, 5, 4}};
    migraphx::shape s_b{migraphx::shape::float_type, {1, 4, 3}};
    migraphx::shape s_bias{migraphx::shape::float_type, {1, 5, 3}};

    auto* mm  = p.get_main_module();
    auto a    = mm->add_parameter("a", s_a);
    auto b    = mm->add_parameter("b", s_b);
    auto bias = mm->add_parameter("bias", s_bias);

    auto* pm  = p.create_module("mlir_dot_add_tanh");
    pm->set_bypass();
    auto px0  = pm->add_parameter("x0", s_a);
    auto px1  = pm->add_parameter("x1", s_b);
    auto px2  = pm->add_parameter("x2", s_bias);
    auto dot  = pm->add_instruction(migraphx::make_op("dot"), px0, px1);
    auto add  = pm->add_instruction(migraphx::make_op("add"), dot, px2);
    auto tanh = pm->add_instruction(migraphx::make_op("tanh"), add);
    if(dot_has_multiple_outputs)
        pm->add_instruction(migraphx::make_op("mul"), dot, px2);
    pm->add_return({tanh});

    return add_fused_mlir_op(p, *pm, {a, b, bias});
}

static migraphx::instruction_ref wrap_mlir_for_gpu_compile(migraphx::program& p,
                                                           migraphx::instruction_ref mlir_ins)
{
    auto* mm = p.get_main_module();
    auto alloc = mm->add_instruction(migraphx::make_op(
        "allocate", {{"shape", migraphx::to_value(mlir_ins->get_shape())}}));
    std::vector<migraphx::instruction_ref> inputs = mlir_ins->inputs();
    inputs.push_back(alloc);
    return mm->insert_instruction(
        mm->end(),
        migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(mlir_ins->get_operator())}}),
        inputs,
        mlir_ins->module_inputs());
}

static void compile_fused_mlir_with_solution(migraphx::program& p,
                                             migraphx::instruction_ref mlir_ins,
                                             const char* perf_config)
{
    migraphx::gpu::context ctx;
    auto ins = wrap_mlir_for_gpu_compile(p, mlir_ins);
    migraphx::gpu::compile(ctx, ins, mlir_ins->get_operator(), perf_config);
}

TEST_CASE(dot_add_tanh_split_k_reports_non_fusible)
{
    if(not mlir_enabled())
        test::skip("MLIR is not enabled");

    migraphx::program p;
    auto mlir_ins = build_dot_add_tanh_mlir(p);

    migraphx::gpu::context ctx;
    const auto& fused_mod = *mlir_ins->module_inputs().front();
    EXPECT(migraphx::gpu::is_module_fusible(fused_mod, ctx, split_k1_perf_config));
    EXPECT(not migraphx::gpu::is_module_fusible(fused_mod, ctx, split_k4_perf_config));
}

TEST_CASE(dot_add_tanh_split_k_compiles_without_split_crash)
{
    if(not mlir_enabled())
        test::skip("MLIR is not enabled");
    if(not migraphx::gpu::has_compiler_for("gpu::mlir_op"))
        test::skip("gpu::mlir_op compiler is unavailable");

    migraphx::program p;
    auto mlir_ins = build_dot_add_tanh_mlir(p);

    // Before the find_final_split guard this path could SIGSEGV when rocMLIR reports the fused
    // module as non-fusible for splitKFactor > 1.
    compile_fused_mlir_with_solution(p, mlir_ins, split_k4_perf_config);
}

TEST_CASE(dot_multiple_outputs_split_k_compiles_without_split_crash)
{
    if(not mlir_enabled())
        test::skip("MLIR is not enabled");
    if(not migraphx::gpu::has_compiler_for("gpu::mlir_op"))
        test::skip("gpu::mlir_op compiler is unavailable");

    migraphx::program p;
    auto mlir_ins = build_dot_add_tanh_mlir(p, true);

    migraphx::gpu::context ctx;
    const auto& fused_mod = *mlir_ins->module_inputs().front();
    EXPECT(not migraphx::gpu::is_module_fusible(fused_mod, ctx, split_k4_perf_config));

    // When dot has multiple consumers, get_output_path(dot) has length 1 and adjacent_find would
    // previously dereference end().
    compile_fused_mlir_with_solution(p, mlir_ins, split_k4_perf_config);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
