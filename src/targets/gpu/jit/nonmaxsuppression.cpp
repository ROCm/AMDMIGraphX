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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>

#include <algorithm>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// nms_data is laid out as { float score; float box[4]; int box_index; } for a
// total of 24 bytes per entry. The scratch workspace is allocated as raw int8
// and reinterpreted in the kernel.
static constexpr std::size_t nms_bytes_per_data = 24;

// Phase 1 ("sort") kernel: each block normalizes its (batch, class)'s boxes
// and bitonic-sorts them by descending score into a per-block region of the
// `sorted` scratch buffer. Launch dimensions are sized to AlignedNumBoxes so
// the sort has enough parallelism even when NumBoxes is small relative to it.
// NOLINTNEXTLINE
static const char* const nms_load_sort_kernel_src = R"__migraphx__(
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
// used, so per-block contents are deterministic.
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
                                auto counts,
                                auto raw_out) {
        nonmaxsuppression_filter<${num_batches},
                                 ${num_classes},
                                 ${num_boxes},
                                 ${aligned_num_boxes}>(
            sorted, max_p, iou_p, thr_p, mask, counts, raw_out);
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
    make_tensors()(${args})([](auto bc_counts, auto output_indices, auto output_num_selected) {
        nonmaxsuppression_compact<${num_batch_class}, ${num_boxes}>(bc_counts, output_indices, output_num_selected);
    });
}

}

} // namespace migraphx
)__migraphx__";

// TODO: use compute_block_size and/or compute_global_for?
// TODO: Don't need num_batches, num_classes, num_boxes as template parameters since tensor_view has shapes.
struct nms_compiler : compiler<nms_compiler>
{
    std::vector<std::string> names() const { return {"nonmaxsuppression"}; }

    // Compile the sort kernel.
    // inputs: [boxes, scores, sorted]
    operation
    compile_load_sort(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& boxes_s  = inputs[0];
        const auto& scores_s = inputs[1];
        const auto num_batches        = boxes_s.lens()[0];
        const auto num_boxes          = boxes_s.lens()[1];
        const auto num_classes        = scores_s.lens()[1];
        const auto aligned_b = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(num_boxes)));
        // clamp between 64 and 1024 threads based on aligned_num_boxes
        const auto block_size = std::min<std::size_t>(std::max<std::size_t>(aligned_b, std::size_t{64}), std::size_t{1024});

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "nms_sort_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, block_size * num_batches * num_classes, block_size);

        auto src = interpolate_string(
            nms_sort_kernel_src,
            {{"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"num_batches", std::to_string(num_batches)},
             {"num_classes", std::to_string(num_classes)},
             {"num_boxes", std::to_string(num_boxes)},
             {"aligned_num_boxes", std::to_string(aligned_b)},
             {"center_point_box",
              v.at("center_point_box").to<bool>() ? "true" : "false"}});
        return compile_hip_code_object(ctx, src, options);
    }

    // inputs: [sorted, max, iou, score_thr, mask, counts, raw_output]
    // `raw_output` is the last input so the framework treats it as the(
    // kernel's chained output flowing into the compact kernel. The filter's
    // inner loops are O(N) per (batch, class), so the launch is sized to
    // NumBoxes (not AlignedNumBoxes) to avoid leaving padding-only threads
    // idle. nb, nc, b are passed through the augmented value because the
    // filter's inputs no longer carry the raw boxes / scores shapes.
    operation
    compile_filter(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto nb        = v.at("num_batches").to<std::size_t>();
        const auto nc        = v.at("num_classes").to<std::size_t>();
        const auto b         = v.at("num_boxes").to<std::size_t>();
        const auto aligned_b = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));

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
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "nms_filter_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, block_size * nb * nc, block_size);

        auto src = interpolate_string(
            nms_filter_kernel_src,
            {{"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"num_batches", std::to_string(nb)},
             {"num_classes", std::to_string(nc)},
             {"num_boxes", std::to_string(b)},
             {"aligned_num_boxes", std::to_string(aligned_b)}});
        return compile_hip_code_object(ctx, src, options);
    }

    // TODO: REDO this whole thing. It doesn't make sense.
    // Compiles the nms_compact_kernel.
    operation
    compile_compact(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // Derive num_blocks (length of counts) and per-block stride NumBoxes
        // (raw_output is sized nb*nc*NumBoxes*3 int64 entries).
        const auto& cnt_s     = inputs[0];
        const auto& raw_s     = inputs[1];
        const auto num_blocks = cnt_s.elements();
        const auto num_boxes  = (num_blocks > 0)
                                    ? raw_s.elements() / (num_blocks * std::size_t{3})
                                    : std::size_t{0};

        const auto total = std::max(num_blocks * num_boxes, std::size_t{1});
        const auto block_size = std::min<std::size_t>(
            std::max<std::size_t>(
                static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(total))),
                std::size_t{64}),
            std::size_t{256});

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "nms_compact_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, 1, block_size);

        auto src = interpolate_string(
            nms_compact_kernel_src,
            {{"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"num_batch_class", std::to_string(num_batch_class)},
             {"num_boxes", std::to_string(num_boxes)}});
        return compile_hip_code_object(ctx, src, options);
    }

    // Required compiler<> hook, should not be used for this compiler.
    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        return {};
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        // ins->inputs() is [user_inputs..., output_alloc] from
        // insert_precompile_op. user_inputs has 2..5 entries per ONNX NMS.
        auto raw = ins->inputs();
        if(raw.size() < 3 or raw.size() > 6)
            MIGRAPHX_THROW("nms_compiler: unexpected input count " + std::to_string(raw.size()));

        std::vector<shape> raw_shapes;
        raw_shapes.reserve(raw.size() - 1);
        std::transform(raw.begin(),
                       raw.end() - 1,
                       std::back_inserter(raw_shapes),
                       [](auto i) { return i->get_shape(); });

        // Default shapes for missing optional scalar inputs. The literals
        // inserted by the replace lambda use these same shapes so the
        // compiled kernel's tensor_view types match the runtime arguments.
        const shape default_max_s{shape::int64_type, {1}};
        const shape default_iou_s{shape::float_type, {1}};
        const shape default_thr_s{shape::float_type, {1}};
        if(raw_shapes.size() < 3)
            raw_shapes.push_back(default_max_s);
        if(raw_shapes.size() < 4)
            raw_shapes.push_back(default_iou_s);
        if(raw_shapes.size() < 5)
            raw_shapes.push_back(default_thr_s);

        const auto& boxes_s  = raw_shapes[0];
        const auto& scores_s = raw_shapes[1];
        if(boxes_s.lens().size() != 3 or scores_s.lens().size() != 3)
            MIGRAPHX_THROW("nms_compiler: boxes and scores must be 3-D");

        const auto nb         = boxes_s.lens()[0];
        const auto b          = boxes_s.lens()[1];
        const auto nc         = scores_s.lens()[1];
        const auto aligned_b  = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));
        const auto iou_packed = (b > 1) ? (b * (b - 1) / 2) : std::size_t{1};

        shape sorted_shape{shape::int8_type, {nb * nc * aligned_b * nms_bytes_per_data}};
        shape mask_shape{shape::uint8_type, {nb * nc * iou_packed}};
        // Per-block output: nb*nc blocks, each can write up to b
        // selections of (batch, class, box_idx) int64 triples.
        shape output_s{shape::int64_type, {nb * nc * b * 3}};
        // Per-batch-per-class selection counts (one index_int per (batch, class) block).
        shape bc_counts_shape{shape::int32_type, {nb * nc}};

        // Sort kernel input shapes:   [boxes, scores, sorted]
        std::vector<shape> sort_shapes = {boxes_s, scores_s, sorted_shape};

        // Filter kernel input shapes: [sorted, max, iou, thr, mask, counts, raw_out]
        std::vector<shape> filter_shapes = {sorted_shape,
                                            raw_shapes[2],
                                            raw_shapes[3],
                                            raw_shapes[4],
                                            mask_shape,
                                            bc_counts_shape,
                                            raw_output_s};

        std::vector<shape> compact_shapes = {bc_counts_shape, output_s, {shape::int64_type, {1}}};

        // The filter kernel can't recover nb/nc/b from its input shapes
        // (sorted/mask/counts/raw_out are all flat scratch buffers), so we
        // pass them through an augmented value alongside the op attributes.
        value augmented        = op.to_value();
        augmented["num_batches"] = nb;
        augmented["num_classes"] = nc;
        augmented["num_boxes"]   = b;

        auto sort_kop    = compile_sort(ctx, sort_shapes, augmented);
        auto filter_kop  = compile_filter(ctx, filter_shapes, augmented);
        auto compact_kop = compile_compact(ctx, compact_shapes, augmented);

        // kernel operations
        std::vector<operation> kops = {sort_kop, filter_kop, compact_kop};

        return {kops,
                [=](module& m, instruction_ref rep_ins, const std::vector<operation>& ops) {
                    auto args = rep_ins->inputs();
                    auto output  = args.back();
                    args.pop_back();
                    
                    // fill out optional arguments
                    if(args.size() < 3)
                    {
                        args.push_back(m.insert_literal(
                            rep_ins, literal{default_max_s, {std::int64_t{0}}}));
                    }
                    if(args.size() < 4)
                    {
                        args.push_back(
                            m.insert_literal(rep_ins, literal{default_iou_s, {0.0f}}));
                    }
                    if(args.size() < 5)
                    {
                        args.push_back(
                            m.insert_literal(rep_ins, literal{default_thr_s, {0.0f}}));
                    }

                    auto sorted = m.insert_instruction(rep_ins, make_op("hip::allocate", {{"shape", to_value(sorted_shape)}}));
                    auto mask = m.insert_instruction(rep_ins, make_op("hip::allocate", {{"shape", to_value(mask_shape)}}));
                    auto bc_counts = m.insert_instruction(rep_ins, make_op("hip::allocate", {{"shape", to_value(bc_counts_shape)}}));
                    auto output_num_selected = m.insert_instruction(rep_ins, make_op("hip::allocate", {{"shape", to_value(scalar_shape)}}));

                    auto load_sort_ins = m.insert_instruction(rep_ins, ops[0], {args[0], args[1], sorted});

                    auto filter_ins = m.insert_instruction(
                        rep_ins,
                        ops[1],
                        {load_sort_ins, args[2], args[3], args[4], mask, bc_counts, output});

                    output = m.insert_instruction(rep_ins, make_op("get_tuple_elem", {{"index", 0}}), filter_ins); 
                    auto bc_counts_output = m.insert_instruction(rep_ins, make_op("get_tuple_elem", {{"index", 1}}), filter_ins);
                    m.replace_instruction(rep_ins, ops[2], {bc_counts_output, output, output_num_selected});
                }};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
