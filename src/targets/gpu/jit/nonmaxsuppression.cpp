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

// NOLINTNEXTLINE
static const char* const nms_kernel_src = R"__migraphx__(
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
                                auto count,
                                auto out) {
        nonmaxsuppression<${center_point_box},
                          ${num_batches},
                          ${num_classes},
                          ${num_boxes},
                          ${aligned_num_boxes}>(
            boxes, scores, max_p, iou_p, thr_p, sorted, mask, count, out);
    });
}

}

} // namespace migraphx
)__migraphx__";

struct nms_compiler : compiler<nms_compiler>
{
    std::vector<std::string> names() const { return {"nonmaxsuppression"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // inputs (in order): boxes, scores, max, iou, score_thr,
        //                    sorted_data, iou_mask, global_count, output.
        const auto& boxes_s   = inputs[0];
        const auto& scores_s  = inputs[1];
        const auto nb         = boxes_s.lens()[0];
        const auto b          = boxes_s.lens()[1];
        const auto nc         = scores_s.lens()[1];
        const auto aligned_b  = static_cast<std::size_t>(bit_ceil(static_cast<std::uint64_t>(b)));
        const auto block_size = std::min<std::size_t>(aligned_b, std::size_t{1024});

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "nms_kernel";
        options.virtual_inputs = inputs;
        options.set_launch_params(v, block_size * nb * nc, block_size);

        auto src = interpolate_string(
            nms_kernel_src,
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
        shape count_s{shape::int64_type, {1}};

        std::vector<shape> kshapes = raw_shapes;
        kshapes.push_back(sorted_s);
        kshapes.push_back(mask_s);
        kshapes.push_back(count_s);
        kshapes.push_back(raw.back()->get_shape());

        auto kop = compile_op(ctx, kshapes, op.to_value());

        return {kop, [=](module& m, instruction_ref ins2, const operation& cop) {
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
                    auto count = m.insert_instruction(
                        ins2, make_op("hip::allocate", {{"shape", to_value(count_s)}}));

                    // Reset the global atomic counter to zero each launch and
                    // pre-zero the output buffer so unwritten rows match the
                    // CPU implementation's behavior.
                    count = m.insert_instruction(
                        ins2, make_op("hip::fill", {{"value", 0}}), count);
                    out = m.insert_instruction(
                        ins2, make_op("hip::fill", {{"value", 0}}), out);

                    args.push_back(sorted);
                    args.push_back(mask);
                    args.push_back(count);
                    args.push_back(out);

                    m.replace_instruction(ins2, cop, args);
                }};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
