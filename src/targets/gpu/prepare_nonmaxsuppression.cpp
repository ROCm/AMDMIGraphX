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
 *
 */
#include <migraphx/gpu/prepare_nonmaxsuppression.hpp>
#include <migraphx/bit.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/register_op.hpp>

#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Sort boxes per (batch, class) into nms_data{} tensor.
struct nms_sort
{
    bool center_point_box = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.center_point_box, "center_point_box"));
    }

    std::string name() const { return "gpu::nms_sort"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        const auto& boxes_s  = inputs.at(0);
        const auto& scores_s = inputs.at(1);
        if(boxes_s.lens().size() != 3 or scores_s.lens().size() != 3)
            MIGRAPHX_THROW("gpu::nms_sort: boxes and scores must be 3-D");
        const auto num_batches = boxes_s.lens()[0];
        const auto num_boxes  = boxes_s.lens()[1];
        const auto num_classes = scores_s.lens()[1];
        const auto aligned_b = static_cast<std::size_t>(bit_ceil(static_cast<std::uint32_t>(num_boxes)));
        shape out_scores_shape{shape::float_type, {num_batches * num_classes, aligned_b}};
        shape out_boxes_shape{shape::float_type, {num_batches * num_classes, aligned_b, 4}};
        shape out_box_index_shape{shape::int32_type, {num_batches * num_classes, aligned_b}};
        return shape{{out_scores_shape, out_boxes_shape, out_box_index_shape}};
    }
};
MIGRAPHX_REGISTER_OP(nms_sort);

// Build the IoU mask and run the greedy filter.
// Produces a tuple of (raw_output, bc_counts).
// num_batches/num_classes/num_boxes are kept as op attributes because the filter inputs
// is a scratch buffer from which these can't be recovered.
struct nms_filter
{
    std::size_t num_batches = 0;
    std::size_t num_classes = 0;
    std::size_t num_boxes   = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.num_batches, "num_batches"),
                    f(self.num_classes, "num_classes"),
                    f(self.num_boxes, "num_boxes"));
    }

    std::string name() const { return "gpu::nms_filter"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        shape output_shape{shape::int64_type, {num_batches * num_classes, num_boxes, 3}};
        shape bc_counts_shape{shape::int32_type, {num_batches * num_classes}};
        return shape{{output_shape, bc_counts_shape}};
    }
};
MIGRAPHX_REGISTER_OP(nms_filter);

// TODO: This should work in-place, saving memory. Need to update IR to handle it.
//  Needs a make_tuple type of operator that reuses the indicies input.
// Prefix-scan the per-block counts and compact the selections into
// the final selected_indices. Output as selected_indices and num_selected tuple.
struct nms_compact
{
    std::string name() const { return "gpu::nms_compact"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        const auto& raw_out_s    = inputs.at(1);
        const auto max_num_boxes = raw_out_s.elements() / std::size_t{3};
        shape selected_indices_shape{shape::int64_type, {max_num_boxes, 3}};
        shape num_selected_shape{shape::int64_type, {1}};
        return shape{{selected_indices_shape, num_selected_shape}};
    }
};
MIGRAPHX_REGISTER_OP(nms_compact);

namespace {

struct find_nonmaxsuppression
{
    auto matcher() const { return match::name("nonmaxsuppression"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto inputs = ins->inputs();
        if(inputs.size() < 2 or inputs.size() > 5)
            MIGRAPHX_THROW("prepare_nonmaxsuppression: unexpected input count " +
                           std::to_string(inputs.size()));

        const auto& boxes_s  = inputs[0]->get_shape();
        const auto& scores_s = inputs[1]->get_shape();
        if(boxes_s.ndim() != 3 or scores_s.ndim() != 3)
            MIGRAPHX_THROW("prepare_nonmaxsuppression: boxes and scores must be 3-D");

        const auto num_batches = boxes_s.lens()[0];
        const auto num_boxes = boxes_s.lens()[1];
        const auto num_classes = scores_s.lens()[1];
        const auto iou_packed = (num_boxes * (num_boxes - 1) / 2);

        // Fill in missing optional scalar inputs with default literals.
        // TODO: this is the wrong way to handle this. Should be checking if the input is eval'able.
        const shape default_max_s{shape::int64_type, {1}};
        const shape default_iou_s{shape::float_type, {1}};
        const shape default_thr_s{shape::float_type, {1}};
        if(inputs.size() < 3)
            inputs.push_back(m.insert_literal(ins, literal{default_max_s, {std::int64_t{0}}}));
        if(inputs.size() < 4)
            inputs.push_back(m.insert_literal(ins, literal{default_iou_s, {0.0f}}));
        if(inputs.size() < 5)
            inputs.push_back(m.insert_literal(ins, literal{default_thr_s, {0.0f}}));

        auto op_val           = ins->get_operator().to_value();
        bool center_point_box = op_val.at("center_point_box").to<bool>();

        // Mask is scratch only; allocate up-front so the standard
        // replace_allocate pass can later turn it into hip::allocate.
        shape mask_shape{shape::uint8_type, {num_batches * num_classes, iou_packed}};
        auto mask_alloc =
            m.insert_instruction(ins, make_op("allocate", {{"shape", to_value(mask_shape)}}));

        auto sorted = m.insert_instruction(
            ins,
            make_op("gpu::nms_sort", {{"center_point_box", center_point_box}}),
            inputs[0],
            inputs[1]);

        auto filter = m.insert_instruction(
            ins,
            make_op("gpu::nms_filter",
                    {{"num_batches", num_batches}, {"num_classes", num_classes}, {"num_boxes", num_boxes}}),
            sorted,
            inputs[2],
            inputs[3],
            inputs[4],
            mask_alloc);

        auto raw_output =
            m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 0}}), filter);
        auto bc_counts =
            m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 1}}), filter);

        auto compact =
            m.insert_instruction(ins, make_op("gpu::nms_compact"), bc_counts, raw_output);

        m.replace_instruction(ins, compact);
    }
};

} // namespace

void prepare_nonmaxsuppression::apply(module& m) const
{
    match::find_matches(m, find_nonmaxsuppression{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
