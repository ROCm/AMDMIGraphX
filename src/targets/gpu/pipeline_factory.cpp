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
#include <migraphx/gpu/pipeline_factory.hpp>
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/check_context.hpp>
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
#include <migraphx/layout_convolution.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/optimize_module.hpp>
#include <migraphx/preallocate_param.hpp>
#include <migraphx/promote_literals.hpp>
#include <migraphx/propagate_precision.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/rewrite_dot.hpp>
#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/rewrite_low_precision.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/rewrite_resize.hpp>
#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/rewrite_topk.hpp>
#include <migraphx/schedule.hpp>
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/simplify_reshapes.hpp>
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
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/sync_device.hpp>
#include <migraphx/gpu/write_literals.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_SCHEDULE_PASS)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_NHWC)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_REWRITE_DOT)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_REWRITE_LRN)
#ifndef _WIN32
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_CK)
#endif
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_FULL_DYNAMIC)

context* pipeline_factory::get_context() const
{
    return any_cast<context>(gctx_ptr);
}

// clang-format off
std::vector<pass> pipeline_factory::dynamic_shapes_pipeline() const
{
    return {
            enable_pass(disabled(MIGRAPHX_ENABLE_FULL_DYNAMIC{}), split_single_dyn_dim{}),
            dead_code_elimination{},
            simplify_dyn_ops{},
            dead_code_elimination{}
    };
}

std::vector<pass> pipeline_factory::required_pipeline() const
{
    return {
            normalize_ops{},
            dead_code_elimination{},
            eliminate_identity{},
            dead_code_elimination{},
            enable_pass(not gpu::gfx_has_fp8ocp_intrinsics() and gpu::gfx_has_fp8fnuz_intrinsics(), fp8_ocp_to_fnuz{}),
            enable_pass(not gpu::gfx_has_fp8ocp_intrinsics() and gpu::gfx_has_fp8fnuz_intrinsics(), dead_code_elimination{}),
            simplify_qdq{.use_mx_quant=gpu::gfx_has_mx_intrinsics()},
            enable_pass(not mlir_enabled(), rewrite_quantization{}),
            dead_code_elimination{},
            rewrite_rnn{},
            dead_code_elimination{},
            eliminate_data_type_for_gpu{.disable_64bit = options.compile_mode != compile_modes::EAGER and options.fast_math},
            rewrite_resize{.affine_only = true},
            dead_code_elimination{}
    };
}

std::vector<pass> pipeline_factory::optimize_rewrite_pipeline() const
{
    return {
            simplify_reshapes{.enable_gather_rewrite = true},
            eliminate_identity{},
            eliminate_pad{},
            dead_code_elimination{},
            insert_pad{{"convolution"}},
            dead_code_elimination{},
            inline_module{},
            enable_pass(disabled(MIGRAPHX_ENABLE_FULL_DYNAMIC{}), rewrite_pooling{.rewrite_lrn = (not MIGRAPHX_USE_MIOPEN or enabled(MIGRAPHX_REWRITE_LRN{}))}),
            dead_code_elimination{},
            rewrite_gelu{options.fast_math},
            optimize_module{},
            layout_convolution{.channels_last = enabled(MIGRAPHX_ENABLE_NHWC{})},
            dead_code_elimination{}
    };
}

std::vector<pass> pipeline_factory::prefuse_pipeline() const
{
    return {
            fuse_horizontal{},
            dead_code_elimination{},
            prefuse_ops{get_context()},
            dead_code_elimination{}
    };
}

std::vector<pass> pipeline_factory::rewrite_simplify_pipeline() const
{
    return {
            rewrite_reduce{},
            rewrite_topk{},
            rewrite_low_precision{},
            enable_pass(enabled(MIGRAPHX_ENABLE_REWRITE_DOT{}), rewrite_dot{}),
            dead_code_elimination{},
            propagate_precision{},
            dead_code_elimination{},
            simplify_reshapes{.enable_op_shape_transform_op=true},
            dead_code_elimination{}
    };
}

std::vector<pass> pipeline_factory::fusion_pipeline() const
{
    return {
            enable_pass(mlir_enabled(), fuse_attention{.attn_enabled = mlir_attention_enabled(get_context()),
                                                .flash_decoding_enabled = mlir_flash_decoding_enabled()}),
            dead_code_elimination{},
            optimize_module{},
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
            auto_contiguous{},
            dead_code_elimination{}
    };
}

std::vector<pass> pipeline_factory::backend_pipeline() const
{
    return {
            lowering{get_context(), options.offload_copy},
            eliminate_contiguous{"gpu::contiguous"},
            dead_code_elimination{},
            adjust_allocation{gpu_allocation_model{.use_hip_allocate = false}},
            dead_code_elimination{},
            eliminate_concat{concat_gpu_optimization{}},
            dead_code_elimination{},
    #if MIGRAPHX_USE_MIOPEN
            compile_miopen{gctx_ptr},
            dead_code_elimination{},
    #endif
            fuse_ops{get_context(), options.fast_math},
            dead_code_elimination{},
    #if MIGRAPHX_USE_HIPBLASLT
            compile_hipblaslt{gctx_ptr},
            dead_code_elimination{},
    #endif
            replace_allocate{gpu_allocation_model{}, options.offload_copy},
            dead_code_elimination{},
            adjust_allocation{gpu_allocation_model{}},
            dead_code_elimination{},
            compile_ops{get_context(), options.exhaustive_tune, options.compile_mode == compile_modes::EAGER},
            dead_code_elimination{},
            promote_literals{},
            dead_code_elimination{},
            write_literals{get_context()},
            schedule{gpu::schedule_model{get_context()->get_current_device().nstreams()}, not enabled(MIGRAPHX_DISABLE_SCHEDULE_PASS{})},
            memory_coloring{"hip::allocate"},
            sync_device{},
            preallocate_param{"scratch", gpu_allocation_model{}},
            dead_code_elimination{},
            eliminate_allocation{"hip::allocate"},
            check_context<context>{},
            normalize_ops{},
            dead_code_elimination{},
            eliminate_identity{}
    };
}
// clang-format on

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
