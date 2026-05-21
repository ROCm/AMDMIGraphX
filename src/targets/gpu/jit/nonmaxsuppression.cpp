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

// NOLINTNEXTLINE
static const char* const nms_sort_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_sort_kernel(${params})
{
    make_tensors()(${args})([](auto boxes,
                               auto scores,
                               auto sorted_scores,
                               auto sorted_boxes,
                               auto sorted_box_indices) {
        nonmaxsuppression_sort<${center_point_box},
                               ${num_batches},
                               ${num_classes},
                               ${num_boxes},
                               ${aligned_num_boxes}>(
           boxes,
           scores,
           sorted_scores,
           sorted_boxes,
           sorted_box_indices);
    });
}

}

} // namespace migraphx
)__migraphx__";

// NOLINTNEXTLINE
static const char* const nms_filter_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_filter_kernel(${params})
{
    make_tensors()(${args})([](auto sorted_scores,
                               auto sorted_boxes,
                               auto sorted_box_indices,
                               auto max_p,
                               auto iou_p,
                               auto thr_p,
                               auto mask,
                               auto output,
                               auto counts) {
        nonmaxsuppression_filter<${num_batches},
                                 ${num_classes},
                                 ${num_boxes},
                                 ${aligned_num_boxes}>(
            sorted_scores,
            sorted_boxes,
            sorted_box_indices,
            max_p,
            iou_p,
            thr_p,
            mask,
            output,
            counts);
    });
}

}

} // namespace migraphx
)__migraphx__";

// NOLINTNEXTLINE
static const char* const nms_compact_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_compact_kernel(${params})
{
    make_tensors()(${args})([](const auto bc_counts,
                               auto indices,
                               auto selected_indices,
                               auto num_selected) {
        nonmaxsuppression_compact<${num_batch_class}, ${num_boxes}>(
            bc_counts,
            indices,
            selected_indices,
            num_selected);
    });
}

}

} // namespace migraphx
)__migraphx__";

// `inputs` is the precompile_op input list:  [boxes, scores, sorted_alloc].
struct nms_sort_compiler : compiler<nms_sort_compiler>
{
    std::vector<std::string> names() const { return {"gpu::nms_sort"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& boxes_s    = inputs[0];
        const auto& scores_s   = inputs[1];
        const auto num_batches = boxes_s.lens()[0];
        const auto num_boxes   = boxes_s.lens()[1];
        const auto num_classes = scores_s.lens()[1];
        const auto aligned_num_boxes =
            static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(num_boxes)));
        // NOTE: topK kernel uses relement/4 for amount of work in a block?
        auto block_size = compute_block_size(ctx, aligned_num_boxes, 1024);

        hip_compile_options options;
        options.inputs         = flatten_shapes(inputs);
        options.output         = inputs.back();
        options.kernel_name    = "nms_sort_kernel";
        options.virtual_inputs = options.inputs;
        options.set_launch_params(v, num_batches * num_classes * block_size, block_size);

        auto src = interpolate_string(
            nms_sort_kernel_src,
            {{"params", enum_params(options.inputs.size(), "void * private_p")},
             {"args", enum_params(options.inputs.size(), "private_p")},
             {"num_batches", std::to_string(num_batches)},
             {"num_classes", std::to_string(num_classes)},
             {"num_boxes", std::to_string(num_boxes)},
             {"aligned_num_boxes", std::to_string(aligned_num_boxes)},
             {"center_point_box", v.at("center_point_box").to<bool>() ? "true" : "false"}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

// `inputs` is the precompile_op input list: [sorted, max, iou, thr, mask, tuple_alloc].
// Where `tuple_alloc` is a tuple allocation holding (raw_output, bc_counts).
// After flattening the tuple, the kernel sees 7 arguments.
struct nms_filter_compiler : compiler<nms_filter_compiler>
{
    std::vector<std::string> names() const { return {"gpu::nms_filter"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto num_batches = v.at("num_batches").to<std::size_t>();
        const auto num_classes = v.at("num_classes").to<std::size_t>();
        const auto num_boxes   = v.at("num_boxes").to<std::size_t>();
        const auto aligned_num_boxes =
            static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(num_boxes)));
        // TODO: tune for max block size?
        // ceil_div(num_boxes, 2) because of strided thread work distribution
        const auto block_size = compute_block_size(ctx, (num_boxes + 1) / 2, 256);

        hip_compile_options options;
        options.inputs         = flatten_shapes(inputs);
        options.output         = inputs.back();
        options.kernel_name    = "nms_filter_kernel";
        options.virtual_inputs = options.inputs;
        options.set_launch_params(v, num_batches * num_classes * block_size, block_size);

        auto src =
            interpolate_string(nms_filter_kernel_src,
                               {{"params", enum_params(options.inputs.size(), "void * private_p")},
                                {"args", enum_params(options.inputs.size(), "private_p")},
                                {"num_batches", std::to_string(num_batches)},
                                {"num_classes", std::to_string(num_classes)},
                                {"num_boxes", std::to_string(num_boxes)},
                                {"aligned_num_boxes", std::to_string(aligned_num_boxes)}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

// `inputs` is the precompile_op input list: [bc_counts, raw_output, tuple_alloc]
// where `tuple_alloc` is a tuple allocation holding (selected_indices, num_selected).
// After flattening, the kernel sees 4 arguments.
struct nms_compact_compiler : compiler<nms_compact_compiler>
{
    std::vector<std::string> names() const { return {"gpu::nms_compact"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& cnt_s          = inputs[0];
        const auto& indices_s      = inputs[1];
        const auto num_batch_class = cnt_s.elements();
        const auto num_boxes       = indices_s.elements() / (num_batch_class * std::size_t{3});
        // TODO: tune for block size?
        // num_boxes block size could also work?
        const auto block_size = compute_block_size(ctx, num_batch_class * num_boxes, 256);

        hip_compile_options options;
        options.inputs         = flatten_shapes(inputs);
        options.output         = inputs.back();
        options.kernel_name    = "nms_compact_kernel";
        options.virtual_inputs = options.inputs;
        options.set_launch_params(v, block_size, block_size);

        auto src =
            interpolate_string(nms_compact_kernel_src,
                               {{"params", enum_params(options.inputs.size(), "void * private_p")},
                                {"args", enum_params(options.inputs.size(), "private_p")},
                                {"num_batch_class", std::to_string(num_batch_class)},
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
