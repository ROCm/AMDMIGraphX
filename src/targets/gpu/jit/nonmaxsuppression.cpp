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

// Phase-1 ("compute") kernel: each block runs NMS for its (batch, class) and
// writes selections into a per-block region of the raw_output scratch plus a
// per-block count. No global atomic counter is used, so per-block contents
// are deterministic.
// NOLINTNEXTLINE
static const char* const nms_compute_kernel_src = R"__migraphx__(
#include <migraphx/kernels/nonmaxsuppression.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void nms_kernel(${params})
{
    make_tensors()(${args})([](auto boxes,
                                auto scores,
                                auto max_p,
                                auto iou_p,
                                auto thr_p,
                                auto sorted,
                                auto mask,
                                auto counts,
                                auto raw_out) {
        nonmaxsuppression<${center_point_box},
                          ${num_batches},
                          ${num_classes},
                          ${num_boxes},
                          ${aligned_num_boxes}>(
            boxes, scores, max_p, iou_p, thr_p, sorted, mask, counts, raw_out);
    });
}

}

} // namespace migraphx
)__migraphx__";

// Phase-2 ("compact") kernel: a single thread walks the per-block raw_output
// regions in block_id order and copies the first counts[b] selections from
// each region into a contiguous prefix of the final output. The order of
// (block_id 0, 1, ...) is the same as the CPU op's (batch, class) iteration
// order, so the resulting output matches the CPU op exactly.
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

    // Compile the per-block compute kernel. `inputs` is:
    //   [boxes, scores, max, iou, score_thr, sorted, mask, counts, raw_output]
    // `raw_output` is the last input so the framework treats it as the
    // kernel's output buffer; the per-block counts is an in/out scratch.
    operation
    compile_compute(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& boxes_s   = inputs[0];
        const auto& scores_s  = inputs[1];
        const auto nb         = boxes_s.lens()[0];
        const auto b          = boxes_s.lens()[1];
        const auto nc         = scores_s.lens()[1];
        const auto aligned_b  = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));
        const auto block_size = std::min<std::size_t>(aligned_b, std::size_t{1024});

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back(); // raw_output buffer
        options.kernel_name    = "nms_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, block_size * nb * nc, block_size);

        auto src = interpolate_string(
            nms_compute_kernel_src,
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

    // Compile the serial compaction kernel. `inputs` is:
    //   [counts, raw_output, output]
    // Launched with one thread (single block, single thread) since the work
    // is intentionally serial: it walks per-block regions in fixed order to
    // produce the exact byte-for-byte output the CPU op produces.
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

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "nms_compact_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, std::size_t{1}, std::size_t{1});

        auto src = interpolate_string(
            nms_compact_kernel_src,
            {{"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"num_blocks", std::to_string(num_blocks)},
             {"num_boxes", std::to_string(num_boxes)}});
        return compile_hip_code_object(ctx, src, options);
    }

    // Required compiler<> hook: return the compute kernel based on the raw
    // input shapes. The full two-kernel chain is handled in `compile()`; this
    // entry point is only used by callers that ask for a single op view.
    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        return compile_compute(ctx, inputs, v);
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

        // Compute kernel input shapes: [user inputs..., sorted, mask, counts, raw_out]
        std::vector<shape> compute_shapes = raw_shapes;
        compute_shapes.push_back(sorted_s);
        compute_shapes.push_back(mask_s);
        compute_shapes.push_back(counts_s);
        compute_shapes.push_back(raw_output_s);

        // Compact kernel input shapes: [counts, raw_out, output]
        std::vector<shape> compact_shapes;
        compact_shapes.push_back(counts_s);
        compact_shapes.push_back(raw_output_s);
        compact_shapes.push_back(raw.back()->get_shape());

        auto compute_kop = compile_compute(ctx, compute_shapes, op.to_value());
        auto compact_kop = compile_compact(ctx, compact_shapes, op.to_value());

        std::vector<operation> kops = {compute_kop, compact_kop};

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

                    auto compute_args = args;
                    compute_args.push_back(sorted);
                    compute_args.push_back(mask);
                    compute_args.push_back(counts);
                    compute_args.push_back(raw_out);

                    auto compute_ins =
                        m.insert_instruction(ins2, cops[0], compute_args);

                    // Use compute_ins (returned raw_out) as the dataflow edge
                    // so the compact kernel is ordered after the compute
                    // kernel and the raw_out buffer remains live.
                    std::vector<instruction_ref> compact_args = {
                        counts, compute_ins, out};
                    m.replace_instruction(ins2, cops[1], compact_args);
                }};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
