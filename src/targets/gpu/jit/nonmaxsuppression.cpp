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
#include <migraphx/bit.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/instruction.hpp>

#include <algorithm>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Phase-1 ("sort") kernel: each block normalizes its (batch, class)'s boxes
// and bitonic-sorts them by descending score into a per-block region of the
// `sorted` scratch buffer. Launch dimensions are sized to AlignedNumBoxes so
// the sort has enough parallelism even when NumBoxes is small relative to it.
// NOLINTNEXTLINE
static const char* const nms_sort_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_sort_kernel(${params})
{
    make_tensors()(${args})([](auto boxes, auto scores, auto sorted) {
        nonmaxsuppression_sort<${center_point_box},
                               ${num_batches},
                               ${num_classes},
                               ${num_boxes},
                               ${aligned_num_boxes}>(boxes, scores, sorted);
    });
}

}

} // namespace migraphx
)__migraphx__";

// Phase-2 ("filter") kernel: each block reads its (batch, class)'s sorted
// records out of the shared `sorted` buffer, builds the IoU mask, runs the
// greedy filter, and writes selections into a per-block region of the
// `raw_output` scratch plus a per-block count. No global atomic counter is
// used, so per-block contents are deterministic. The argument order after the
// `mask` scratch reflects the precompile_op tuple output flatten order:
// (raw_output, bc_counts).
// NOLINTNEXTLINE
static const char* const nms_filter_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_filter_kernel(${params})
{
    make_tensors()(${args})([](auto sorted,
                                auto max_p,
                                auto iou_p,
                                auto thr_p,
                                auto mask,
                                auto raw_out,
                                auto counts) {
        nonmaxsuppression_filter<${num_batches},
                                 ${num_classes},
                                 ${num_boxes},
                                 ${aligned_num_boxes}>(
            sorted, max_p, iou_p, thr_p, mask, raw_out, counts);
    });
}

}

} // namespace migraphx
)__migraphx__";

// Phase-3 ("compact") kernel: a single block does an exclusive prefix scan
// over the per-block counts to obtain output offsets, then its threads
// scatter selections from each per-block region of `raw_output` into the
// contiguous prefix of the final output. The order of (block_id 0, 1, ...)
// is the same as the CPU op's (batch, class) iteration order, so the
// resulting output matches the CPU op exactly.
// NOLINTNEXTLINE
static const char* const nms_compact_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_compact_kernel(${params})
{
    make_tensors()(${args})([](auto bc_counts,
                               auto raw_output,
                               auto output_indices,
                               auto output_num_selected) {
        nonmaxsuppression_compact<${num_batch_class}, ${num_boxes}>(
            bc_counts, raw_output, output_indices, output_num_selected);
    });
}

}

} // namespace migraphx
)__migraphx__";

// Compiler for the per-(batch, class) sort kernel. `inputs` is the
// precompile_op input list:  [boxes, scores, sorted_alloc].
struct nms_sort_compiler : compiler<nms_sort_compiler>
{
    std::vector<std::string> names() const { return {"gpu::nms_sort"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& boxes_s  = inputs[0];
        const auto& scores_s = inputs[1];
        const auto nb        = boxes_s.lens()[0];
        const auto b         = boxes_s.lens()[1];
        const auto nc        = scores_s.lens()[1];
        const auto aligned_b =
            static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));
        // Clamp the block size to [64, 1024] threads, sized for the bitonic sort.
        const auto block_size = std::min<std::size_t>(
            std::max<std::size_t>(aligned_b, std::size_t{64}), std::size_t{1024});

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "nms_sort_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, block_size * nb * nc, block_size);

        auto src = interpolate_string(
            nms_sort_kernel_src,
            {{"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"num_batches", std::to_string(nb)},
             {"num_classes", std::to_string(nc)},
             {"num_boxes", std::to_string(b)},
             {"aligned_num_boxes", std::to_string(aligned_b)},
             {"center_point_box", v.at("center_point_box").to<bool>() ? "true" : "false"}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

// Compiler for the filter kernel. `inputs` is the precompile_op input list:
//   [sorted, max, iou, thr, mask, tuple_alloc]
// where `tuple_alloc` is a tuple allocation holding (raw_output, bc_counts).
// After flattening the tuple, the kernel sees 7 arguments.
struct nms_filter_compiler : compiler<nms_filter_compiler>
{
    std::vector<std::string> names() const { return {"gpu::nms_filter"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto nb = v.at("num_batches").to<std::size_t>();
        const auto nc = v.at("num_classes").to<std::size_t>();
        const auto b  = v.at("num_boxes").to<std::size_t>();
        const auto aligned_b =
            static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));

        // Clamp the per-block thread count to [64, 256]: a multiple of the
        // wavefront size keeps __syncthreads / block_scan well-defined, and
        // 256 is the sweet spot for the O(N) inner loops without inflating
        // shared-memory pressure on `removed[N]` (which is sized by N, not by
        // block_size).
        const auto block_size = std::min<std::size_t>(
            std::max<std::size_t>(
                static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b))),
                std::size_t{64}),
            std::size_t{256});

        hip_compile_options options;
        options.inputs         = flatten(inputs);
        options.output         = inputs.back();
        options.kernel_name    = "nms_filter_kernel";
        options.virtual_inputs = options.inputs;
        options.set_launch_params(v, block_size * nb * nc, block_size);

        auto src = interpolate_string(
            nms_filter_kernel_src,
            {{"params", enum_params(options.inputs.size(), "void * private_p")},
             {"args", enum_params(options.inputs.size(), "private_p")},
             {"num_batches", std::to_string(nb)},
             {"num_classes", std::to_string(nc)},
             {"num_boxes", std::to_string(b)},
             {"aligned_num_boxes", std::to_string(aligned_b)}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

// Compiler for the compact kernel. `inputs` is the precompile_op input list:
//   [bc_counts, raw_output, tuple_alloc]
// where `tuple_alloc` is a tuple allocation holding (selected_indices,
// num_selected). After flattening, the kernel sees 4 arguments. `num_blocks`
// (a.k.a. nb*nc) and `num_boxes` are recovered from the input shapes.
struct nms_compact_compiler : compiler<nms_compact_compiler>
{
    std::vector<std::string> names() const { return {"gpu::nms_compact"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& cnt_s     = inputs[0];
        const auto& raw_s     = inputs[1];
        const auto num_blocks = cnt_s.elements();
        const auto num_boxes  = (num_blocks > 0)
                                    ? raw_s.elements() / (num_blocks * std::size_t{3})
                                    : std::size_t{0};

        const auto total      = std::max(num_blocks * num_boxes, std::size_t{1});
        const auto block_size = std::min<std::size_t>(
            std::max<std::size_t>(
                static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(total))),
                std::size_t{64}),
            std::size_t{256});

        hip_compile_options options;
        options.inputs         = flatten(inputs);
        options.output         = inputs.back();
        options.kernel_name    = "nms_compact_kernel";
        options.virtual_inputs = options.inputs;
        options.set_launch_params(v, block_size, block_size);

        auto src = interpolate_string(
            nms_compact_kernel_src,
            {{"params", enum_params(options.inputs.size(), "void * private_p")},
             {"args", enum_params(options.inputs.size(), "private_p")},
             {"num_batch_class", std::to_string(num_blocks)},
             {"num_boxes", std::to_string(num_boxes)}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
