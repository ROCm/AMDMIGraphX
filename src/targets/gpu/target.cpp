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
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/check_context.hpp>
#include <migraphx/convert_to_json.hpp>
#include <migraphx/compile_modes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/fp8_ocp_to_fnuz.hpp>
#include <migraphx/fuse_attention.hpp>
#include <migraphx/fuse_concat.hpp>
#include <migraphx/fuse_horizontal.hpp>
#include <migraphx/fuse_pointwise_reduce.hpp>
#include <migraphx/inline_module.hpp>
#include <migraphx/insert_pad.hpp>
#include <migraphx/json.hpp>
#include <migraphx/layout_convolution.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/optimize_module.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/preallocate_param.hpp>
#include <migraphx/promote_literals.hpp>
#include <migraphx/promote_storage_type.hpp>
#include <migraphx/propagate_precision.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/rewrite_convolution.hpp>
#include <migraphx/rewrite_dot.hpp>
#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/rewrite_low_precision.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/rewrite_resize.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/rewrite_topk.hpp>
#include <migraphx/schedule.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/split_reduce.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/gpu/compile_hipblaslt.hpp>
#include <migraphx/gpu/compile_miopen.hpp>
#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/concat_gpu_opt.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/eliminate_data_type_for_gpu.hpp>
#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/lower_device_ops.hpp>
#include <migraphx/gpu/lower_reshape.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/sync_device.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SCHEDULE_PASS)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_OPTIONS)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_REWRITE_DOT)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_REWRITE_LRN)
#ifndef _WIN32
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_CK)
#endif
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SET_GEMM_PROVIDER)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_FULL_DYNAMIC)

namespace {
// Backend options recognized by the GPU target, supplied via
// compile_options::backend_options.
struct backend_options
{
    std::vector<std::string> mlss_use_specific_ops = {};
    // Read/write problem caches (the common case: a user tuning a model). New
    // tuning solutions are saved back to these files.
    std::vector<std::string> problem_cache_files = {};
    // Read-only problem caches (system-level, e.g. shipped by gpuep or an ISV),
    // searched after the writable caches and never written back.
    std::vector<std::string> read_only_problem_cache_files = {};
    // Layout used for convolutions, by name: channels_first, channels_last, or channels_auto.
    layout_convolution::layout_order convolution_layout = layout_convolution::channels_auto;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mlss_use_specific_ops, "mlss_use_specific_ops"),
                    f(self.problem_cache_files, "problem_cache_files"),
                    f(self.read_only_problem_cache_files, "read_only_problem_cache_files"),
                    f(self.convolution_layout, "convolution_layout"));
    }
};

// The backend options passed to compile, with any key set by MIGRAPHX_GPU_OPTIONS (a json-like
// object such as "{convolution_layout:channels_last}") overriding it.
backend_options get_backend_options(const compile_options& options)
{
    auto opts = options.backend_options;
    auto env  = string_value_of(MIGRAPHX_GPU_OPTIONS{});
    if(not env.empty())
    {
        auto v = from_json_string(convert_to_json(env));
        if(not v.is_object())
            MIGRAPHX_THROW("MIGRAPHX_GPU_OPTIONS must be a json object");
        for(const auto& opt : v)
            opts[opt.get_key()] = opt.without_key();
    }
    return from_value<backend_options>(value(opts));
}

struct pipeline_factory
{
    migraphx::context* gctx_ptr = nullptr;
    compile_options options;
    backend_options backend_opts = {};

    migraphx::context* get_generic_context() const { return gctx_ptr; }

    // cppcheck-suppress CastIntegerToAddressAtReturn
    context* get_context() const { return any_cast<context>(gctx_ptr); }

    std::vector<pass> dynamic_shapes_pipeline() const
    {
        return {
            enable_pass(disabled(MIGRAPHX_ENABLE_FULL_DYNAMIC{}), split_single_dyn_dim{}),
            dead_code_elimination{},
            simplify_dyn_ops{},
            dead_code_elimination{},
        };
    }

    std::vector<pass> required_pipeline() const
    {
        return {
            normalize_ops{},
            dead_code_elimination{},
            eliminate_identity{},
            dead_code_elimination{},
            enable_pass(not gpu::gfx_has_fp8ocp_intrinsics(*get_context()) and
                            gpu::gfx_has_fp8fnuz_intrinsics(*get_context()),
                        fp8_ocp_to_fnuz{}),
            enable_pass(not gpu::gfx_has_fp8ocp_intrinsics(*get_context()) and
                            gpu::gfx_has_fp8fnuz_intrinsics(*get_context()),
                        dead_code_elimination{}),
            simplify_qdq{.use_mx_quant = gpu::gfx_has_mx_intrinsics(*get_context())},
            enable_pass(not mlir_enabled(), rewrite_quantization{}),
            dead_code_elimination{},
            eliminate_data_type_for_gpu{.disable_64bit = options.fast_math, .ctx = get_context()},
            rewrite_resize{.affine_only = true},
            dead_code_elimination{},
            simplify_reshapes{.enable_gather_rewrite = true},
            eliminate_identity{},
            eliminate_pad{},
            dead_code_elimination{},
            insert_pad{{"convolution"}},
            dead_code_elimination{},
            inline_module{},
            enable_pass(disabled(MIGRAPHX_ENABLE_FULL_DYNAMIC{}),
                        rewrite_pooling{.rewrite_lrn = (not MIGRAPHX_USE_MIOPEN or
                                                        enabled(MIGRAPHX_REWRITE_LRN{}))}),
            dead_code_elimination{},
        };
    }

    std::vector<pass> optimize_rewrite_pipeline() const
    {
        auto gfx_name          = get_context()->get_current_device().get_gfx_name();
        bool bf16_missing_valu = not starts_with(gfx_name, "gfx125");
        return {
            rewrite_convolution{},
            dead_code_elimination{},
            rewrite_gelu{options.fast_math},
            optimize_module{},
            layout_convolution{.order = backend_opts.convolution_layout},
            dead_code_elimination{},
            enable_pass(disabled(MIGRAPHX_ENABLE_FULL_DYNAMIC{}), fuse_horizontal{}),
            dead_code_elimination{},
            prefuse_ops{get_context()},
            dead_code_elimination{},
            dead_code_elimination{},
            rewrite_reduce{},
            rewrite_topk{},
            rewrite_low_precision{},
            enable_pass(enabled(MIGRAPHX_ENABLE_REWRITE_DOT{}), rewrite_dot{}),
            dead_code_elimination{},
            enable_pass(bf16_missing_valu, promote_storage_type{{shape::bf16_type}}),
            dead_code_elimination{},
            propagate_precision{},
            dead_code_elimination{},
            simplify_reshapes{.enable_op_shape_transform_op = true},
            dead_code_elimination{},
        };
    }

    std::vector<pass> fusion_pipeline() const
    {
        return {
            enable_pass(options.compile_mode != compile_modes::eager and mlir_enabled(),
                        fuse_attention{.attn_enabled = mlir_attention_enabled(get_context()),
                                       .flash_decoding_enabled = mlir_flash_decoding_enabled()}),
            dead_code_elimination{},
            optimize_module{},
            fuse_mlss{.ctx = get_context(), .use_specific_ops = backend_opts.mlss_use_specific_ops},
            fuse_pointwise_reduce{},
            dead_code_elimination{},
#ifndef _WIN32
            enable_pass(enabled(MIGRAPHX_ENABLE_CK{}), fuse_ck{}),
#endif
            dead_code_elimination{},
            enable_pass(mlir_enabled(), fuse_mlir{get_context()}),
            dead_code_elimination{},
            fuse_concat{},
            dead_code_elimination{},
        };
    }

    std::vector<pass> backend_pipeline() const
    {
        std::size_t max_memory =
            get_context()->is_cross_compile() ? std::numeric_limits<std::size_t>::max() : 0;
        return {
            auto_contiguous{},
            dead_code_elimination{},
            lowering{get_context(), options.offload_copy},
            eliminate_contiguous{"gpu::contiguous"},
            dead_code_elimination{},
            lower_reshape{},
            dead_code_elimination{},
            adjust_allocation{gpu_allocation_model{.use_hip_allocate = false}},
            dead_code_elimination{},
            eliminate_concat{concat_gpu_optimization{}},
            dead_code_elimination{},
#if MIGRAPHX_USE_MIOPEN
            compile_miopen{get_generic_context()},
            dead_code_elimination{},
#endif
            fuse_ops{get_context(), options.fast_math},
            dead_code_elimination{},
#if MIGRAPHX_USE_HIPBLASLT
            compile_hipblaslt{get_generic_context()},
            dead_code_elimination{},
#endif
            replace_allocate{gpu_allocation_model{}, options.offload_copy},
            dead_code_elimination{},
            adjust_allocation{gpu_allocation_model{}},
            dead_code_elimination{},
            lower_device_ops{},
            compile_ops{get_context(),
                        options.exhaustive_tune,
                        options.compile_mode == compile_modes::eager},
            dead_code_elimination{},
            promote_literals{},
            dead_code_elimination{},
            write_literals{.max_memory = max_memory},
            schedule{gpu::schedule_model{get_context()->get_current_device().nstreams()},
                     not enabled(MIGRAPHX_DISABLE_SCHEDULE_PASS{})},
            memory_coloring{"hip::allocate"},
            sync_device{},
            preallocate_param{"scratch", gpu_allocation_model{}},
            dead_code_elimination{},
            eliminate_allocation{"hip::allocate"},
            check_context<context>{},
            normalize_ops{},
            dead_code_elimination{},
            eliminate_identity{},
        };
    }
};
} // namespace

std::vector<pass> target::get_passes(migraphx::context& gctx, const compile_options& options) const
{
    auto& ctx = any_cast<context>(gctx);
    ctx.set_exhaustive_tune_flag(options.exhaustive_tune);

    if(options.compile_mode == compile_modes::max)
        ctx.set_exhaustive_tune_flag(true);

    auto backend_opts = get_backend_options(options);

    // Problem cache files arrive as GPU backend options. The writable caches
    // (problem_cache_files) save new tuning solutions back; the read-only caches
    // (read_only_problem_cache_files) are system-level and never written.
    ctx.load_problem_caches(backend_opts.read_only_problem_cache_files,
                            backend_opts.problem_cache_files);

    pipeline_factory p{&gctx, options, backend_opts};

    std::vector<std::vector<pass>> pipelines;

    if(options.compile_mode == compile_modes::eager)
    {
        pipelines = {
            p.dynamic_shapes_pipeline(),
            p.required_pipeline(),
            {optimize_module{},
             dead_code_elimination{},
             rewrite_reduce{},
             rewrite_topk{},
             dead_code_elimination{}},
            p.fusion_pipeline(),
            p.backend_pipeline(),
        };
    }
    else
    {
        pipelines = {
            p.dynamic_shapes_pipeline(),
            p.required_pipeline(),
            p.optimize_rewrite_pipeline(),
            p.fusion_pipeline(),
            p.backend_pipeline(),
        };
    }

    std::vector<pass> passes;
    std::copy(pipelines.begin(), pipelines.end(), join_back_inserter(passes));
    return passes;
}

std::string target::name() const { return "gpu"; }

migraphx::context target::get_context() const
{
    if(is_cross_compile())
        return context(desc);
    return context(gpu::get_device_id());
}

argument target::copy_to(const argument& arg) const
{
    if(is_cross_compile())
        MIGRAPHX_THROW("Cannot copy data in cross-compilation mode");
    return gpu::to_gpu(arg);
}

argument target::copy_from(const argument& arg) const
{
    if(is_cross_compile())
        MIGRAPHX_THROW("Cannot copy data in cross-compilation mode");
    return gpu::from_gpu(arg);
}

argument target::allocate(const shape& s) const
{
    if(is_cross_compile())
        MIGRAPHX_THROW("Cannot allocate GPU memory in cross-compilation mode");
    return gpu::allocate_gpu(s);
}

MIGRAPHX_REGISTER_TARGET(target);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
