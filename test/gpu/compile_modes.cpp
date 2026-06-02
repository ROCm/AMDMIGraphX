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
#include <migraphx/compile_modes.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/register_target.hpp>
#include <test.hpp>
#include <algorithm>
#include <string>
#include <vector>

static std::vector<std::string> get_pass_names(migraphx::compile_modes mode)
{
    migraphx::gpu::target tgt;
    auto ctx = tgt.get_context();
    migraphx::compile_options options;
    options.compile_mode = mode;
    auto passes = tgt.get_passes(ctx, options);
    std::vector<std::string> names;
    std::transform(passes.begin(), passes.end(), std::back_inserter(names), [](const auto& p) {
        return p.name();
    });
    return names;
}

static bool contains_pass(const std::vector<std::string>& names, const std::string& name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

static std::size_t count_pass(const std::vector<std::string>& names, const std::string& name)
{
    return std::count(names.begin(), names.end(), name);
}

// All passes that only exist in optimize_rewrite_pipeline (not in any other pipeline).
// These must be absent from eager mode and present in balanced mode.
static std::vector<std::string> optimize_rewrite_only_passes()
{
    return {
        "rewrite_gelu",
        "layout_convolution",
        "gpu::prefuse_ops",
        "rewrite_reduce",
        "rewrite_topk",
        "rewrite_low_precision",
        "propagate_precision",
    };
}

TEST_CASE(balanced_mode_pass_names)
{
    auto names = get_pass_names(migraphx::compile_modes::balanced);

    // --- dynamic_shapes_pipeline ---
    // split_single_dyn_dim or id (conditional)
    EXPECT(contains_pass(names, "split_single_dyn_dim") or contains_pass(names, "id"));
    EXPECT(contains_pass(names, "simplify_dyn_ops"));

    // --- required_pipeline ---
    EXPECT(contains_pass(names, "normalize_ops"));
    EXPECT(contains_pass(names, "eliminate_identity"));
    // fp8_ocp_to_fnuz or id (conditional on GPU arch)
    EXPECT(contains_pass(names, "simplify_qdq"));
    // rewrite_quantization or id (conditional on mlir)
    EXPECT(contains_pass(names, "rewrite_rnn"));
    EXPECT(contains_pass(names, "gpu::eliminate_data_type_for_gpu"));
    EXPECT(contains_pass(names, "rewrite_resize"));
    EXPECT(contains_pass(names, "simplify_reshapes"));
    EXPECT(contains_pass(names, "eliminate_pad"));
    EXPECT(contains_pass(names, "insert_pad"));
    EXPECT(contains_pass(names, "inline_module"));
    // rewrite_pooling or id (conditional)

    // --- optimize_rewrite_pipeline (only in balanced/max, NOT in eager) ---
    EXPECT(contains_pass(names, "rewrite_gelu"));
    EXPECT(contains_pass(names, "layout_convolution"));
    EXPECT(contains_pass(names, "gpu::prefuse_ops"));
    EXPECT(contains_pass(names, "rewrite_reduce"));
    EXPECT(contains_pass(names, "rewrite_topk"));
    EXPECT(contains_pass(names, "rewrite_low_precision"));
    EXPECT(contains_pass(names, "propagate_precision"));
    // fuse_horizontal or id (conditional)
    // rewrite_dot or id (conditional)

    // --- fusion_pipeline ---
    // fuse_attention or id (conditional on mlir)
    EXPECT(contains_pass(names, "optimize_module"));
    EXPECT(contains_pass(names, "fuse_pointwise_reduce"));
    // fuse_ck or id (conditional, non-Windows)
    // fuse_mlir or id (conditional on mlir)
    EXPECT(contains_pass(names, "fuse_concat"));

    // --- backend_pipeline ---
    EXPECT(contains_pass(names, "auto_contiguous"));
    EXPECT(contains_pass(names, "gpu::lowering"));
    EXPECT(contains_pass(names, "eliminate_contiguous"));
    EXPECT(contains_pass(names, "adjust_allocation"));
    EXPECT(contains_pass(names, "eliminate_concat"));
#if MIGRAPHX_USE_MIOPEN
    EXPECT(contains_pass(names, "gpu::compile_miopen"));
#endif
    EXPECT(contains_pass(names, "gpu::fuse_ops"));
#if MIGRAPHX_USE_HIPBLASLT
    EXPECT(contains_pass(names, "gpu::compile_hipblaslt"));
#endif
    EXPECT(contains_pass(names, "replace_allocate"));
    EXPECT(contains_pass(names, "gpu::compile_ops"));
    EXPECT(contains_pass(names, "promote_literals"));
    EXPECT(contains_pass(names, "gpu::write_literals"));
    EXPECT(contains_pass(names, "schedule"));
    EXPECT(contains_pass(names, "memory_coloring"));
    EXPECT(contains_pass(names, "sync_device"));
    EXPECT(contains_pass(names, "preallocate_param"));
    EXPECT(contains_pass(names, "eliminate_allocation"));
    EXPECT(contains_pass(names, "check_context"));

    // optimize_module appears twice in balanced: once in optimize_rewrite, once in fusion
    EXPECT(count_pass(names, "optimize_module") == 2);
}

TEST_CASE(eager_mode_pass_names)
{
    auto names = get_pass_names(migraphx::compile_modes::eager);

    // --- dynamic_shapes_pipeline ---
    EXPECT(contains_pass(names, "split_single_dyn_dim") or contains_pass(names, "id"));
    EXPECT(contains_pass(names, "simplify_dyn_ops"));

    // --- required_pipeline (all present) ---
    EXPECT(contains_pass(names, "normalize_ops"));
    EXPECT(contains_pass(names, "eliminate_identity"));
    EXPECT(contains_pass(names, "simplify_qdq"));
    EXPECT(contains_pass(names, "rewrite_rnn"));
    EXPECT(contains_pass(names, "gpu::eliminate_data_type_for_gpu"));
    EXPECT(contains_pass(names, "rewrite_resize"));
    EXPECT(contains_pass(names, "simplify_reshapes"));
    EXPECT(contains_pass(names, "eliminate_pad"));
    EXPECT(contains_pass(names, "insert_pad"));
    EXPECT(contains_pass(names, "inline_module"));

    // --- optimize_rewrite_pipeline is SKIPPED ---
    // All optimize_rewrite-only passes must be absent
    for(const auto& pass_name : optimize_rewrite_only_passes())
    {
        EXPECT(not contains_pass(names, pass_name));
    }

    // --- fusion_pipeline (all present) ---
    EXPECT(contains_pass(names, "optimize_module"));
    EXPECT(contains_pass(names, "fuse_pointwise_reduce"));
    EXPECT(contains_pass(names, "fuse_concat"));

    // --- backend_pipeline (all present) ---
    EXPECT(contains_pass(names, "auto_contiguous"));
    EXPECT(contains_pass(names, "gpu::lowering"));
    EXPECT(contains_pass(names, "eliminate_contiguous"));
    EXPECT(contains_pass(names, "adjust_allocation"));
    EXPECT(contains_pass(names, "eliminate_concat"));
#if MIGRAPHX_USE_MIOPEN
    EXPECT(contains_pass(names, "gpu::compile_miopen"));
#endif
    EXPECT(contains_pass(names, "gpu::fuse_ops"));
#if MIGRAPHX_USE_HIPBLASLT
    EXPECT(contains_pass(names, "gpu::compile_hipblaslt"));
#endif
    EXPECT(contains_pass(names, "replace_allocate"));
    EXPECT(contains_pass(names, "gpu::compile_ops"));
    EXPECT(contains_pass(names, "promote_literals"));
    EXPECT(contains_pass(names, "gpu::write_literals"));
    EXPECT(contains_pass(names, "schedule"));
    EXPECT(contains_pass(names, "memory_coloring"));
    EXPECT(contains_pass(names, "sync_device"));
    EXPECT(contains_pass(names, "preallocate_param"));
    EXPECT(contains_pass(names, "eliminate_allocation"));
    EXPECT(contains_pass(names, "check_context"));

    // optimize_module appears only once in eager (from fusion_pipeline only)
    EXPECT(count_pass(names, "optimize_module") == 1);
}

TEST_CASE(eager_vs_balanced_exact_difference)
{
    auto eager_names    = get_pass_names(migraphx::compile_modes::eager);
    auto balanced_names = get_pass_names(migraphx::compile_modes::balanced);

    // Eager must have strictly fewer passes
    EXPECT(eager_names.size() < balanced_names.size());

    // Compute the difference: passes in balanced but not in eager.
    // Since the same pass name can appear multiple times, we work with sorted copies.
    auto eager_sorted    = eager_names;
    auto balanced_sorted = balanced_names;
    std::sort(eager_sorted.begin(), eager_sorted.end());
    std::sort(balanced_sorted.begin(), balanced_sorted.end());

    std::vector<std::string> difference;
    std::set_difference(balanced_sorted.begin(),
                        balanced_sorted.end(),
                        eager_sorted.begin(),
                        eager_sorted.end(),
                        std::back_inserter(difference));

    // The difference should consist exclusively of passes from optimize_rewrite_pipeline.
    // Every pass in the difference must be either a known optimize_rewrite pass,
    // dead_code_elimination (there are extra DCEs in optimize_rewrite), or a conditional
    // pass that resolves to "id".
    auto known_optimize_passes = optimize_rewrite_only_passes();
    for(const auto& name : difference)
    {
        bool is_optimize_rewrite_pass =
            std::find(known_optimize_passes.begin(), known_optimize_passes.end(), name) !=
            known_optimize_passes.end();
        bool is_dce               = (name == "dead_code_elimination");
        bool is_optimize_module   = (name == "optimize_module");
        bool is_simplify_reshapes = (name == "simplify_reshapes");
        bool is_conditional       = (name == "id");
        // fuse_horizontal is conditional but its name is fuse_horizontal when enabled
        bool is_fuse_horizontal = (name == "fuse_horizontal");
        bool is_rewrite_dot     = (name == "rewrite_dot");
        EXPECT(is_optimize_rewrite_pass or is_dce or is_optimize_module or is_simplify_reshapes or
               is_conditional or is_fuse_horizontal or is_rewrite_dot);
    }

    // Verify all optimize_rewrite-only passes appear in the difference
    for(const auto& pass_name : known_optimize_passes)
    {
        bool found =
            std::find(difference.begin(), difference.end(), pass_name) != difference.end();
        EXPECT(found);
    }
}

TEST_CASE(max_mode_matches_balanced_passes)
{
    auto max_names      = get_pass_names(migraphx::compile_modes::max);
    auto balanced_names = get_pass_names(migraphx::compile_modes::balanced);

    // max mode produces the exact same pass list as balanced
    EXPECT(max_names.size() == balanced_names.size());
    EXPECT(max_names == balanced_names);
}

TEST_CASE(max_mode_enables_exhaustive_tune)
{
    migraphx::gpu::target tgt;
    auto ctx = tgt.get_context();
    migraphx::compile_options options;
    options.compile_mode    = migraphx::compile_modes::max;
    options.exhaustive_tune = false;
    tgt.get_passes(ctx, options);

    auto& gpu_ctx = migraphx::any_cast<migraphx::gpu::context>(ctx);
    EXPECT(gpu_ctx.get_exhaustive_tune_flag());
}

TEST_CASE(balanced_mode_does_not_force_exhaustive_tune)
{
    migraphx::gpu::target tgt;
    auto ctx = tgt.get_context();
    migraphx::compile_options options;
    options.compile_mode    = migraphx::compile_modes::balanced;
    options.exhaustive_tune = false;
    tgt.get_passes(ctx, options);

    auto& gpu_ctx = migraphx::any_cast<migraphx::gpu::context>(ctx);
    EXPECT(not gpu_ctx.get_exhaustive_tune_flag());
}

TEST_CASE(eager_mode_does_not_force_exhaustive_tune)
{
    migraphx::gpu::target tgt;
    auto ctx = tgt.get_context();
    migraphx::compile_options options;
    options.compile_mode    = migraphx::compile_modes::eager;
    options.exhaustive_tune = false;
    tgt.get_passes(ctx, options);

    auto& gpu_ctx = migraphx::any_cast<migraphx::gpu::context>(ctx);
    EXPECT(not gpu_ctx.get_exhaustive_tune_flag());
}

TEST_CASE(eager_preserves_pass_order)
{
    auto eager_names    = get_pass_names(migraphx::compile_modes::eager);
    auto balanced_names = get_pass_names(migraphx::compile_modes::balanced);

    // Eager's passes should appear in the same relative order as in balanced.
    // Walk through balanced names and check that eager passes appear in order.
    std::size_t eager_idx = 0;
    for(const auto& bname : balanced_names)
    {
        if(eager_idx >= eager_names.size())
            break;
        if(bname == eager_names[eager_idx])
            eager_idx++;
    }
    // All eager passes should have been matched in order
    EXPECT(eager_idx == eager_names.size());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
