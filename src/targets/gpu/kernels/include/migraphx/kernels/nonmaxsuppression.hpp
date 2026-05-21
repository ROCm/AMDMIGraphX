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
#ifndef MIGRAPHX_GUARD_KERNELS_NONMAXSUPPRESSION_HPP
#define MIGRAPHX_GUARD_KERNELS_NONMAXSUPPRESSION_HPP

#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/scan.hpp>
#include <migraphx/kernels/sort.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

template <class Score, class Box, class Index>
struct nms_data
{
    // holds a copy of data
    Score score;
    array<Box, 4> box;
    Index box_index;
};

// Comparator for sorting nms_data{} (or anything else with a `.score` field).
struct nms_score_greater
{
    template <class T>
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a.score > b.score;
    }
};

// Decode a single box into (xmin, ymin, xmax, ymax) corners.
// Normalize such that [x1, y1] is the bottom left corner.
template <bool CenterPointBox, class Box>
__device__ inline array<typename Box::type, 4> nms_normalize_box(const Box box)
{
    if constexpr(CenterPointBox)
    {
        const auto xc = box[0];
        const auto yc = box[1];
        const auto hw = box[2] * 0.5f;
        const auto hh = box[3] * 0.5f;
        return {xc - hw, yc - hh, xc + hw, yc + hh};
    }
    else
    {
        // ONNX layout: [y1, x1, y2, x2]; corners may be in either order.
        const auto y1   = box[0];
        const auto x1   = box[1];
        const auto y2   = box[2];
        const auto x2   = box[3];
        const auto xmin = min(x1, x2);
        const auto xmax = max(x1, x2);
        const auto ymin = min(y1, y2);
        const auto ymax = max(y1, y2);
        return {xmin, ymin, xmax, ymax};
    }
}

template <class Box, class Threshold>
__device__ inline bool nms_iou_over_threshold(const Box a, const Box b, const Threshold threshold)
{
    const auto left   = max(a[0], b[0]);
    const auto right  = min(a[2], b[2]);
    const auto top    = max(a[1], b[1]);
    const auto bottom = min(a[3], b[3]);
    const auto w      = max(right - left, 0.f);
    const auto h      = max(bottom - top, 0.f);
    const auto inter  = w * h;
    const auto area_a = max(a[2] - a[0], 0.f) * max(a[3] - a[1], 0.f);
    const auto area_b = max(b[2] - b[0], 0.f) * max(b[3] - b[1], 0.f);
    const auto un     = area_a + area_b - inter;
    if(area_a <= 0.f or area_b <= 0.f or un <= 0.f)
        return false;
    return (inter / un) > threshold;
}

// Packed upper-triangular index for j > i within an N x N matrix.
__device__ inline index_int nms_packed_idx(index_int i, index_int j, index_int N)
{
    return (i * N - (i * (i + 1)) / 2) + j - (i + 1);
}

// One block per (batch_idx, class_idx).
// Load data into per-block buffer of nms_data.
// Pads values after N with sentinel values.
// Sorts the nms_data in descending order by score.
// boxes_tv: dims([NumBatches, NumBoxes, 4])
// scores_tv: dims([NumBatches, NumClasses, NumBoxes])
// sorted_scores: output, dims([B, C, AlignedNumBoxes])
// sorted_boxes: output, dims([B, C, AlignedNumBoxes, 4])
// sorted_indices: output, dims([B, C, AlignedNumBoxes])
template <bool CenterPointBox,
          index_int NumBatches,
          index_int NumClasses,
          index_int NumBoxes,
          index_int AlignedNumBoxes,
          class Boxes,
          class Scores,
          class SortedScores,
          class SortedBoxes,
          class SortedIndices>
__device__ void nonmaxsuppression_sort(const Boxes boxes_tv,
                                       const Scores scores_tv,
                                       SortedScores sorted_scores,
                                       SortedBoxes sorted_boxes,
                                       SortedIndices sorted_indices)
{
    static_assert(NumBatches > 0);
    static_assert(NumClasses > 0);
    static_assert(NumBoxes > 0);
    static_assert(AlignedNumBoxes > 0);

    auto idx                 = make_index();
    const index_int block_id = idx.group;
    const int batch_idx      = block_id / NumClasses;
    const int class_idx      = block_id % NumClasses;

    // numpy indexing: scores[batch_idx, class_idx, :]
    const auto my_scores =
        slice_tensor(scores_tv, array<index_int, 3>{batch_idx, class_idx, 0}, slice_axes<2>());

    using scores_type  = typename SortedScores::type;
    using boxes_type   = typename SortedBoxes::type;
    using indices_type = typename SortedIndices::type;
    // Use shared memory for sorting per-block nms_data. Assuming it fits in LDS.
    // TODO: can add a static_assert on needed LDS size
    __shared__
        uninitialized_buffer<nms_data<scores_type, boxes_type, indices_type>, AlignedNumBoxes>
            block_nms_data;
    idx.local_stride(AlignedNumBoxes, [&](auto i) {
        if(i < NumBoxes)
        {
            block_nms_data[i].score = my_scores[i];
            block_nms_data[i].box   = nms_normalize_box<CenterPointBox>(
                slice_tensor(boxes_tv, array<index_int, 3>{batch_idx, i, 0}, slice_axes<2>()));
            block_nms_data[i].box_index = static_cast<int32_t>(i);
        }
        else
        {
            block_nms_data[i].score     = numeric_lowest<scores_type>();
            block_nms_data[i].box       = array<boxes_type, 4>{0.f, 0.f, 0.f, 0.f};
            block_nms_data[i].box_index = -1;
        }
    });
    __syncthreads();

    bitonic_sort{nms_score_greater{}}.template block_sort<AlignedNumBoxes>(idx, block_nms_data);

    // Copy sorted result back to global memory.
    auto block_out_scores =
        slice_tensor(sorted_scores, array<index_int, 2>{block_id, 0}, slice_axes<1>());
    auto block_out_boxes =
        slice_tensor(sorted_boxes, array<index_int, 3>{block_id, 0, 0}, slice_axes<1, 2>());
    auto block_out_indices =
        slice_tensor(sorted_indices, array<index_int, 2>{block_id, 0}, slice_axes<1>());
    idx.local_stride(AlignedNumBoxes, [&](auto i) {
        block_out_scores[i] = block_nms_data[i].score;
        auto out_box_iter   = block_out_boxes.begin_at(array<index_int, 3>{0, i, 0});
        copy(block_nms_data[i].box.begin(), block_nms_data[i].box.end(), out_box_iter);
        block_out_indices[i] = block_nms_data[i].box_index;
    });
}

// Build the packed upper-triangular IoU mask for the NumBoxes nms_data boxes.
// Work is striped such that each thread does a multiple of 2 rows so each does roughly the same
// amount of work regardless of where it falls in the triangle.
// `nms_data`: nms_data nms_data{} tensor
// `mask`: bool mask tensor
template <index_int NumBoxes, class NMSData, class Mask>
__device__ void
nms_make_iou_mask(const index idx, const NMSData nms_data, Mask mask, const float iou_threshold)
{
    static_assert(NumBoxes > 0);
    constexpr index_int half = NumBoxes / 2;

    auto fill_row = [&](index_int i) {
        for(index_int j = i + 1; j < NumBoxes; ++j)
        {
            mask[nms_packed_idx(i, j, NumBoxes)] =
                nms_iou_over_threshold(nms_data[i].box, nms_data[j].box, iou_threshold);
        }
    };

    idx.local_stride(half, [&](auto i) {
        fill_row(i);
        fill_row(NumBoxes - 1 - i);
    });

    // Have thread 0 do middle row if odd NumBoxes
    if constexpr((NumBoxes & 1) != 0 and NumBoxes > 1)
    {
        if(idx.local == 0)
            fill_row(half);
    }
}

// Greedy filter that writes selections into a per-batch per-class region of output.
template <index_int NumBoxes,
          index_int NumClasses,
          class NMSData,
          class Mask,
          class Output,
          class Counts>
__device__ void nms_filter_per_block(const index idx,
                                     const NMSData nms_data,
                                     const Mask mask,
                                     const int max_output,
                                     const float score_thr,
                                     Output block_output,
                                     Counts bc_counts)
{
    static_assert(NumBoxes > 0);
    const index_int block_id = idx.group;
    const int batch_idx      = block_id / NumClasses;
    const int class_idx      = block_id % NumClasses;
    // TODO: use bits for removed mask
    __shared__ uint8_t removed[NumBoxes];
    // Match the ref op: only filter by score when score_threshold > 0.
    const bool do_filter = score_thr > 0.f;
    idx.local_stride(NumBoxes,
                     [&](auto i) { removed[i] = (do_filter and nms_data[i].score < score_thr); });
    __syncthreads();

    index_int output_idx = 0;
    for(index_int i = 0; i < NumBoxes; ++i)
    {
        if(output_idx >= max_output)
        {
            __syncthreads();
            break;
        }
        if(removed[i] == 0)
        {
            if(idx.local == 0)
            {
                array<typename Output::type, 3> tmp = {batch_idx, class_idx, nms_data[i].box_index};
                auto output_iter = block_output.begin_at(array<index_int, 3>{0, output_idx, 0});
                copy(tmp.begin(), tmp.end(), output_iter);
            }
            ++output_idx;
            for(index_int j = i + 1 + idx.local; j < NumBoxes; j += idx.nlocal())
            {
                removed[j] |= mask[nms_packed_idx(i, j, NumBoxes)];
            }
        }
        __syncthreads();
    }

    if(idx.local == 0)
        bc_counts[block_id] = static_cast<int32_t>(output_idx);
}

// Per-block filter driver: one block per (batch_idx, class_idx).`.
// Expecting box-coordinate convention has already been normalized into corner form.
// TODO: Merge the nonmaxsuppression_sort and nonmaxsuppression_filter kernels by relaxing
// the AlignedNumBoxes resitriction for the sort.
template <index_int NumBatches,
          index_int NumClasses,
          index_int NumBoxes,
          index_int AlignedNumBoxes,
          class SortedScores,
          class SortedBoxes,
          class SortedIndices,
          class MaxOut,
          class IouThr,
          class ScoreThr,
          class Mask,
          class Output,
          class Counts>
__device__ void nonmaxsuppression_filter(const SortedScores sorted_scores,
                                         const SortedBoxes sorted_boxes,
                                         const SortedIndices sorted_indices,
                                         const MaxOut max_out_p,
                                         const IouThr iou_thr_p,
                                         const ScoreThr score_thr_p,
                                         Mask mask,
                                         Output output,
                                         Counts bc_counts)
{
    static_assert(NumBatches > 0);
    static_assert(NumClasses > 0);
    static_assert(NumBoxes > 0);

    auto idx                  = make_index();
    const index_int block_idx = idx.group;

    auto my_sorted_scores =
        slice_tensor(sorted_scores, array<index_int, 2>{block_idx, 0}, slice_axes<1>());
    auto my_sorted_boxes =
        slice_tensor(sorted_boxes, array<index_int, 3>{block_idx, 0, 0}, slice_axes<1, 2>());
    auto my_sorted_indices =
        slice_tensor(sorted_indices, array<index_int, 2>{block_idx, 0}, slice_axes<1>());

    using scores_type  = typename SortedScores::type;
    using boxes_type   = typename SortedBoxes::type;
    using indices_type = typename SortedIndices::type;
    // Use shared memory for sorting per-block nms_data. Assuming it fits in LDS.
    // TODO: can add a static_assert on needed LDS size
    __shared__ uninitialized_buffer<nms_data<scores_type, boxes_type, indices_type>, NumBoxes>
        block_nms_data;

    idx.local_stride(NumBoxes, [&](auto i) {
        block_nms_data[i].score = my_sorted_scores[i];
        auto boxes_iter         = my_sorted_boxes.begin_at(array<index_int, 3>{0, i, 0});
        copy(boxes_iter, boxes_iter + 4, block_nms_data[i].box.begin());
        block_nms_data[i].box_index = my_sorted_indices[i];
    });
    auto my_mask   = slice_tensor(mask, array<index_int, 2>{block_idx, 0}, slice_axes<1>());
    auto my_output = slice_tensor(output, array<index_int, 3>{block_idx, 0, 0}, slice_axes<1, 2>());

    // Read scalar tensor inputs
    const int max_output_boxes_per_class = max_out_p[0];
    const float iou_thr_val              = iou_thr_p[0];
    const float score_thr_val            = score_thr_p[0];

    __syncthreads();
    nms_make_iou_mask<NumBoxes>(idx, block_nms_data, my_mask, iou_thr_val);

    __syncthreads();
    nms_filter_per_block<NumBoxes, NumClasses>(idx,
                                               block_nms_data,
                                               my_mask,
                                               max_output_boxes_per_class,
                                               score_thr_val,
                                               my_output,
                                               bc_counts);
}

// Move batch/class box index entries to the beginning of the output buffer. Runs with 1 block.
// `bc_counts`: Number of selected boxes per batch per class. (read-only)
// `indices`: Box indices, kernel packs selected boxes in-place to the beginning
// of the buffer in (batch, class) iteration order.
// `num_selected`: Total number of selected boxes.
template <index_int NumBatchClass,
          index_int NumBoxes,
          class Counts,
          class Idx,
          class Num,
          class Out>
__device__ void
nonmaxsuppression_compact(const Counts bc_counts, const Idx indices, Out output, Num num_selected)
{
    static_assert(NumBatchClass > 0);
    static_assert(NumBoxes > 0);
    // TODO: get a better bound on this
    static_assert(NumBatchClass <= 8192,
                  "nms_compact: NumBatchClass exceeds the LDS budget for offsets[]");

    auto idx = make_index();
    __shared__ index_int offsets[NumBatchClass];
    // Exclusive prefix sum on bc_counts to get offsets
    // TODO: there's probably a better way to get the exclusive prefix sum rather than doing the
    // minus each time.
    block_scan(
        idx,
        op::sum{},
        0,
        NumBatchClass,
        [&](auto i) -> int32_t { return bc_counts[i]; },
        [&](auto i, auto inclusive_value) { offsets[i] = inclusive_value - bc_counts[i]; });
    __syncthreads();

    // Get num_selected_boxes from last value of exclusive scan and add last bc_counts value.
    if(idx.local == 0)
    {
        num_selected[0] = offsets[NumBatchClass - 1] + bc_counts[NumBatchClass - 1];
    }

    // rearrange index values to make the output packed.
    // TODO: this could be done in-place to save memory.
    constexpr index_int index_size  = 3;
    constexpr index_int max_entries = NumBatchClass * NumBoxes;
    idx.local_stride(max_entries, [&](auto i) {
        const index_int batch_class_idx = i / NumBoxes;
        const index_int box_idx         = i % NumBoxes;
        if(box_idx < bc_counts[batch_class_idx])
        {
            for(int k = 0; k < 3; ++k)
            {
                output[(offsets[batch_class_idx] + box_idx) * index_size + k] =
                    indices[(batch_class_idx * NumBoxes + box_idx) * index_size + k];
            }
        }
    });
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_NONMAXSUPPRESSION_HPP
