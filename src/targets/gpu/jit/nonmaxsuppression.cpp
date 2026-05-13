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
    make_tensors()(${args})([](auto counts, auto raw_out, auto out) {
        nonmaxsuppression_compact<${num_blocks}, ${num_boxes}>(counts, raw_out, out);
    });
}

}

} // namespace migraphx
)__migraphx__";

struct nms_compiler : compiler<nms_compiler>
{
    std::vector<std::string> names() const { return {"nonmaxsuppression"}; }

    // Compile the per-block sort kernel. `inputs` is:
    //   [boxes, scores, sorted]
    // `sorted` is the last input so the framework treats it as the kernel's
    // chained output flowing into the filter kernel. Launch is sized to
    // AlignedNumBoxes so the bitonic sort has enough lane-parallelism even
    // when NumBoxes is small relative to it.
    operation
    compile_sort(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& boxes_s  = inputs[0];
        const auto& scores_s = inputs[1];
        const auto nb        = boxes_s.lens()[0];
        const auto b         = boxes_s.lens()[1];
        const auto nc        = scores_s.lens()[1];
        const auto aligned_b = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));
        // bitonic block_sort uses __syncthreads between every stage; pad up
        // to a wavefront so degenerate cases (e.g. NumBoxes <= 1) still
        // launch a valid block.
        const auto block_size = std::min<std::size_t>(
            std::max<std::size_t>(aligned_b, std::size_t{64}), std::size_t{1024});

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back(); // sorted buffer
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
             {"center_point_box",
              v.at("center_point_box").to<bool>() ? "true" : "false"}});
        return compile_hip_code_object(ctx, src, options);
    }

    // Compile the per-block filter kernel. `inputs` is:
    //   [sorted, max, iou, score_thr, mask, counts, raw_output]
    // `raw_output` is the last input so the framework treats it as the
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
        options.output         = inputs.back(); // raw_output buffer
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

    // Compile the compaction kernel. `inputs` is:
    //   [counts, raw_output, output]
    // Launched as a single block: an exclusive prefix scan over counts gives
    // each per-block region a base offset, then the block's threads scatter
    // selections to those offsets in parallel. The single-block constraint
    // keeps the scan in shared memory; `nms_compact` static_asserts a hard
    // cap on NumBlocks that comfortably fits any realistic ONNX NMS.
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

        // Pick a block size large enough to give the scan and scatter useful
        // parallelism without inflating LDS pressure. block_scan requires the
        // block size to be a multiple of the wavefront size; 64 is the
        // smallest safe choice for all supported gfx targets.
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
        options.set_launch_params(v, block_size, block_size); // one block

        auto src = interpolate_string(
            nms_compact_kernel_src,
            {{"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"num_blocks", std::to_string(num_blocks)},
             {"num_boxes", std::to_string(num_boxes)}});
        return compile_hip_code_object(ctx, src, options);
    }

    // Required compiler<> hook: return the sort kernel built from the raw
    // user input shapes (boxes, scores). The full three-kernel chain is
    // handled in `compile()`; this entry point is only used by callers that
    // ask for a single op view.
    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.size() < 2)
            MIGRAPHX_THROW("nms_compiler: compile_op needs at least boxes and scores");
        const auto& boxes_s  = inputs[0];
        const auto& scores_s = inputs[1];
        if(boxes_s.lens().size() != 3 or scores_s.lens().size() != 3)
            MIGRAPHX_THROW("nms_compiler: boxes and scores must be 3-D");
        const auto nb        = boxes_s.lens()[0];
        const auto b         = boxes_s.lens()[1];
        const auto nc        = scores_s.lens()[1];
        const auto aligned_b = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));
        const shape sorted_s{shape::int8_type, {nb * nc * aligned_b * nms_bytes_per_data}};
        return compile_sort(ctx, {boxes_s, scores_s, sorted_s}, v);
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

        shape sorted_s{shape::int8_type, {nb * nc * aligned_b * nms_bytes_per_data}};
        shape mask_s{shape::uint8_type, {nb * nc * iou_packed}};
        // Per-block raw output: nb*nc blocks, each can write up to b
        // selections of (batch, class, box_idx) int64 triples.
        shape raw_output_s{shape::int64_type, {nb * nc * b * 3}};
        // Per-block selection counts (one int32 per (batch, class) block).
        shape counts_s{shape::int32_type, {nb * nc}};

        // Sort kernel input shapes:   [boxes, scores, sorted]
        std::vector<shape> sort_shapes = {boxes_s, scores_s, sorted_s};

        // Filter kernel input shapes: [sorted, max, iou, thr, mask, counts, raw_out]
        std::vector<shape> filter_shapes = {sorted_s,
                                            raw_shapes[2],
                                            raw_shapes[3],
                                            raw_shapes[4],
                                            mask_s,
                                            counts_s,
                                            raw_output_s};

        // Compact kernel input shapes: [counts, raw_out, output]
        std::vector<shape> compact_shapes = {counts_s, raw_output_s, raw.back()->get_shape()};

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

        std::vector<operation> kops = {sort_kop, filter_kop, compact_kop};

        return {kops,
                [=](module& m, instruction_ref ins2, const std::vector<operation>& cops) {
                    auto args = ins2->inputs();
                    auto out  = args.back();
                    args.pop_back();

                    if(args.size() < 3)
                    {
                        args.push_back(m.insert_literal(
                            ins2, literal{default_max_s, {std::int64_t{0}}}));
                    }
                    if(args.size() < 4)
                    {
                        args.push_back(
                            m.insert_literal(ins2, literal{default_iou_s, {0.0f}}));
                    }
                    if(args.size() < 5)
                    {
                        args.push_back(
                            m.insert_literal(ins2, literal{default_thr_s, {0.0f}}));
                    }

                    auto sorted = m.insert_instruction(
                        ins2, make_op("hip::allocate", {{"shape", to_value(sorted_s)}}));
                    auto mask = m.insert_instruction(
                        ins2, make_op("hip::allocate", {{"shape", to_value(mask_s)}}));
                    auto raw_out = m.insert_instruction(
                        ins2, make_op("hip::allocate", {{"shape", to_value(raw_output_s)}}));
                    auto counts = m.insert_instruction(
                        ins2, make_op("hip::allocate", {{"shape", to_value(counts_s)}}));

                    // Pre-zero the final output buffer so unwritten rows match
                    // the CPU implementation's behavior (trailing zeros). The
                    // counts and raw_out scratch don't need zeroing: each
                    // block writes its count exactly once and the compact
                    // kernel only reads counts[b] entries from each block.
                    out = m.insert_instruction(
                        ins2, make_op("hip::fill", {{"value", 0}}), out);

                    // Phase 1: sort. Inputs are [boxes, scores, sorted]; the
                    // returned `sort_ins` is the post-write `sorted` buffer
                    // which becomes the filter kernel's first input.
                    auto sort_ins = m.insert_instruction(
                        ins2, cops[0], {args[0], args[1], sorted});

                    // Phase 2: filter. Use `sort_ins` as the dataflow edge so
                    // the filter is ordered after sort and `sorted` stays
                    // live. Returned `filter_ins` is the post-write
                    // `raw_output` buffer fed to compact.
                    auto filter_ins = m.insert_instruction(
                        ins2,
                        cops[1],
                        {sort_ins, args[2], args[3], args[4], mask, counts, raw_out});

                    // Phase 3: compact. Counts/filter_ins/out match the
                    // [counts, raw_output, output] order in compact_shapes.
                    m.replace_instruction(ins2, cops[2], {counts, filter_ins, out});
                }};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
