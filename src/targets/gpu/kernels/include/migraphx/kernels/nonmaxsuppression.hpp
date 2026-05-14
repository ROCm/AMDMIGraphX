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

namespace migraphx {

struct nms_data
{
    float score;
    array<float, 4> box;
    int box_index;
};

// Decode a single box into (xmin, ymin, xmax, ymax) corners.
// Normalize such that [x1, y1] is the bottom left corner
template <bool CenterPointBox, class Box>
__device__ inline array<float, 4> nms_normalize_box(Box box)
{
    if constexpr(CenterPointBox)
    {
        const float xc = box[0];
        const float yc = box[1];
        const float hw = box[2] * 0.5f;
        const float hh = box[3] * 0.5f;
        return {xc - hw, yc - hh, xc + hw, yc + hh};
    }
    else
    {
        // ONNX layout: [y1, x1, y2, x2]; corners may be in either order.
        const float y1   = box[0];
        const float x1   = box[1];
        const float y2   = box[2];
        const float x2   = box[3];
        const float xmin = min(x1, x2);
        const float xmax = max(x1, x2);
        const float ymin = min(y1, y2);
        const float ymax = max(y1, y2);
        return {xmin, ymin, xmax, ymax};
    }
}

template <class Box>
__device__ inline bool
nms_iou_over_threshold(const Box a, Box b, float threshold)
{
    const float left   = max(a[0], b[0]);
    const float right  = min(a[2], b[2]);
    const float top    = max(a[1], b[1]);
    const float bottom = min(a[3], b[3]);
    const float w      = max(right - left, 0.f);
    const float h      = max(bottom - top, 0.f);
    const float inter  = w * h;
    const float area_a = max(a[2] - a[0], 0.f) * max(a[3] - a[1], 0.f);
    const float area_b = max(b[2] - b[0], 0.f) * max(b[3] - b[1], 0.f);
    const float un     = area_a + area_b - inter;
    if(area_a <= 0.f or area_b <= 0.f or un <= 0.f)
        return false;
    return (inter / un) > threshold;
}

// Packed upper-triangular index for j > i within an N x N matrix.
__device__ inline index_int nms_packed_idx(index_int i, index_int j, index_int N)
{
    return (i * N - (i * (i + 1)) / 2) + j - (i + 1);
}

// Comparator for sorting nms_data{}.
struct nms_score_greater
{
    constexpr bool operator()(const nms_data& a, const nms_data& b) const
    {
        return a.score > b.score;
    }
};

// Phase 1
// One block per (batch_idx, class_idx).
// Load data into per-block buffer of nms_data.
// Pads values after N with sentinel values.
// Sorts the nms_data in descending order by score.
// boxes_tv: dims([N, 4]) of float.
// scores_tv: dims([N]) of float.
// sorted_tv: dims([N]) of nms_data{}.
template <bool CenterPointBox,
          index_int NumBatches,
          index_int NumClasses,
          index_int NumBoxes,
          index_int AlignedNumBoxes,
          class Boxes,
          class Scores,
          class Output>
__device__ void nonmaxsuppression_sort(Boxes boxes_tv, Scores scores_tv, Output out_tv)
{
    static_assert(NumBatches > 0, "num_batches must be > 0");
    static_assert(NumClasses > 0, "num_classes must be > 0");

    auto idx = make_index();
    const index_int block_id = idx.group;
    const int batch_idx      = static_cast<int>(block_id / NumClasses);
    const int class_idx      = static_cast<int>(block_id % NumClasses);
    
    constexpr auto block_out_shape = make_shape(index_ints<AlignedNumBoxes>{});
    auto* p = reinterpret_cast<nms_data*>(out_tv.data()) + block_id * AlignedNumBoxes;
    auto block_out_tv = make_tensor_view<nms_data>(p, block_out_shape);

    const auto* boxes_b   = boxes_tv.data() + batch_idx * NumBoxes * 4;
    const auto* scores_bc = scores_tv.data() + (batch_idx * NumClasses + class_idx) * NumBoxes;

    nms_data d;
    idx.local_stride(AlignedNumBoxes, [&](auto i) {
        if(i < NumBoxes)
        {
            d.score     = scores_bc[i];
            d.box       = nms_normalize_box<CenterPointBox>(boxes_b + i * 4);
            d.box_index = static_cast<int>(i);
        }
        else
        {
            // Sentinel: -inf score so it never beats any real entry
            d.score     = -__FLT_MAX__;
            d.box       = array<float, 4>{0.f, 0.f, 0.f, 0.f};
            d.box_index = -1;
        }
        block_out_tv[i] = d;
    });
    __syncthreads();
    bitonic_sort<nms_score_greater>{nms_score_greater{}}.template block_sort<AlignedNumBoxes>(idx, block_out_tv);
}

// Phase 2
// Build the packed upper-triangular IoU mask for the N sorted boxes.
// Work is striped such that each thread does a multiple of 2 rows so each does roughly the same
// amount of work regardless of where it falls in the triangle.
// `sorted`: sorted nms_data{} tensor
// `mask`: bool mask tensor
template <index_int N, class SortedData, class Mask>
__device__ void nms_make_iou_mask(index idx, const SortedData sorted, Mask mask, float iou_threshold)
{
    constexpr index_int half = N / 2;

    auto fill_row = [&](index_int i) {
        for(index_int j = i + 1; j < N; ++j)
        {
            mask[nms_packed_idx(i, j, N)] =
                nms_iou_over_threshold(sorted[i].box, sorted[j].box, iou_threshold) ? 1 : 0;
        }
    };

    idx.local_stride(half, [&](auto i) {
        fill_row(i);
        fill_row(N - 1 - i);
    });

    if constexpr((N & 1) != 0 and N > 1)
    {
        if(idx.local == 0)
            fill_row(half);
    }
}

// Phase 2
// Greedy filter that writes selections into a per-block region of a
// scratch buffer (block_id * N entries) and stores the per-block count.
template <index_int N>
__device__ void nms_filter_per_block(index idx,
                                     const nms_data* sorted,
                                     const uint8_t* mask,
                                     int batch_idx,
                                     int class_idx,
                                     int64_t max_output,
                                     float score_thr,
                                     int64_t* raw_output,    // [num_blocks * N * 3]
                                     int32_t* block_counts)  // [num_blocks]
{
    __shared__ uint8_t removed[N > 0 ? N : 1];
    // Match the CPU op: only filter by score when score_threshold > 0 (the CPU
    // takes the same branch). With a non-positive (or sentinel) threshold, all
    // boxes are kept regardless of sign.
    const bool do_filter = score_thr > 0.f;
    idx.local_stride(N, [&](auto i) {
        removed[i] = (do_filter and sorted[i].score < score_thr) ? 1 : 0;
    });
    __syncthreads();

    const index_int block_id = idx.group;
    int64_t* my_output       = raw_output + block_id * N * 3;

    index_int output_idx = 0;
    for(index_int i = 0; i < N; ++i)
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
                my_output[output_idx * 3 + 0] = batch_idx;
                my_output[output_idx * 3 + 1] = class_idx;
                my_output[output_idx * 3 + 2] = sorted[i].box_index;
            }
            ++output_idx;
            // Update removed[] using row i of the IoU mask. Each thread handles
            // a stride of the row to balance work.
            for(index_int j = i + 1 + idx.local; j < N; j += idx.nlocal())
            {
                removed[j] |= mask[nms_packed_idx(i, j, N)];
            }
        }
        __syncthreads();
    }

    if(idx.local == 0)
        block_counts[block_id] = static_cast<int32_t>(output_idx);
}

// Per-block filter driver: one block per (batch_idx, class_idx). Reads the
// previously-sorted records out of `sorted_buf`, builds the IoU mask in
// `mask_buf`, then runs the greedy filter writing selections into a per-block
// region of `raw_output` and the per-block count into `counts_buf`.
//
// Expecting box-coordinate convention has already been normalized into corner form
// in `sorted_buf`.
//
// The parameter order matches the flatten order of the precompile_op tuple
// output (raw_output, counts). `sorted_buf` and `mask_buf` are scratch inputs
// allocated upstream; `raw_output_buf` and `counts_buf` are the two halves of
// the tuple-typed output buffer.
template <index_int NumBatches,
          index_int NumClasses,
          index_int NumBoxes,
          index_int AlignedNumBoxes,
          class Sorted,
          class MaxOut,
          class IouThr,
          class ScoreThr,
          class Mask,
          class RawOutput,
          class Counts>
__device__ void nonmaxsuppression_filter(Sorted sorted_buf,
                                         MaxOut max_out_p,
                                         IouThr iou_thr_p,
                                         ScoreThr score_thr_p,
                                         Mask mask_buf,
                                         RawOutput raw_output_buf,
                                         Counts counts_buf)
{
    static_assert(NumBatches > 0, "num_batches must be > 0");
    static_assert(NumClasses > 0, "num_classes must be > 0");

    auto idx                            = make_index();
    const index_int block_id            = idx.group;
    const int batch_idx                 = block_id / NumClasses;
    const int class_idx                 = block_id % NumClasses;
    constexpr index_int iou_packed_size = (NumBoxes > 1) ? (NumBoxes * (NumBoxes - 1)) / 2 : 1;

    nms_data* my_sorted =
        reinterpret_cast<nms_data*>(sorted_buf.data()) + block_id * AlignedNumBoxes;
    uint8_t* my_mask = reinterpret_cast<uint8_t*>(mask_buf.data()) + block_id * iou_packed_size;

    // Pull scalar tensor inputs once. They're broadcast to all threads via the
    // common load (each thread reads the same single element).
    const int64_t max_output_boxes_per_class = max_out_p[0];
    const float iou_thr_val   = iou_thr_p[0];
    const float score_thr_val = score_thr_p[0];

    if constexpr(NumBoxes > 1)
    {
        nms_make_iou_mask<NumBoxes>(idx, my_sorted, my_mask, iou_thr_val);
        __syncthreads();
    }

    nms_filter_per_block<NumBoxes>(idx,
                                   my_sorted,
                                   my_mask,
                                   batch_idx,
                                   class_idx,
                                   max_output_boxes_per_class,
                                   score_thr_val,
                                   reinterpret_cast<int64_t*>(raw_output_buf.data()),
                                   reinterpret_cast<int32_t*>(counts_buf.data()));
}


// Phase 3
// Move batch/class box index entries to the beginning of the output buffer.
// Runs with 1 block. Reads from `raw_indices` (the filter kernel's per-block
// output) and writes the compacted selections into `output_indices`.
// `bc_counts`: Number of selected boxes per batch per class. (read-only)
// `raw_indices`: Per-block raw indices written by the filter kernel
// (read-only).
// `output_indices`: Output box indices, packed contiguously at the beginning
// of the buffer in (batch, class) iteration order.
// `output_num_selected`: Total number of selected boxes.
template <index_int NumBatchClass,
          index_int NumBoxes,
          class Counts,
          class RawIndices,
          class IdxOutput,
          class NumOutput>
__device__ void nonmaxsuppression_compact(const Counts bc_counts,
                                          RawIndices raw_indices,
                                          IdxOutput output_indices,
                                          NumOutput output_num_selected)
{
    static_assert(NumBatchClass > 0, "NumBatchClass must be > 0");
    static_assert(NumBatchClass <= 16000, "nms_compact: NumBlocks exceeds the LDS budget for offsets[]");
    __shared__ array<index_int, NumBatchClass> offsets;
    // Exclusive prefix sum on bc_counts to get offsets
    block_scan(
        idx,
        op::sum{},
        0,
        NumBlocks,
        [&](auto i) -> int32_t { return bc_counts[i]; },
        [&](auto i, auto inclusive_value) { offsets[i] = inclusive_value - block_counts[i]; });
    __syncthreads();

    // Get num_selected_boxes from last value of exclusive scan and add last bc_counts value.
    if(idx.local == 0)
    {
        output_num_selected[0] = offsets[NumBatchClass-1] + block_counts[NumBlocks-1];
    }

    // swap index values to make the output packed
    constexpr index_int index_size = 3;
    constexpr index_int max_entries = NumBatchClass * NumBoxes;
    idx.local_stride(max_entries, [&](auto i) {
        const index_int batch_class_idx = i / NumBoxes;
        const index_int box_idx = i & NumBoxes;
        if(box_idx < block_counts[batch_class_idx])
        {
            auto src = [&](auto j){return output_indices[batch_class_idx * NumBoxes + box_idx * index_size + j]};
            auto dst = [&](auto j){return output_indices[(offsets[batch_class_idx] + box_idx) * index_size + j]};
            array<int64_t, 3> tmp_src = {src(0), src(1), src(2)};
            for(int k = 0; k < 3; ++k)
            {
                src(k) = dst(k);
                dst(k) = tmp_src[k];
            }
        }
    });
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_NONMAXSUPPRESSION_HPP
