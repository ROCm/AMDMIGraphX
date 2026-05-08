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
#include <migraphx/kernels/sort.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/types.hpp>

namespace migraphx {

// Per-box record carried through the sort. Box corners are stored normalized
// to (xmin, ymin, xmax, ymax) so the IoU computation is independent of the
// center_point_box attribute.
struct nms_data
{
    float score;
    array<float, 4> box;
    int box_index;
};

// Decode a single box into (xmin, ymin, xmax, ymax) corners.
template <bool CenterPointBox>
__device__ inline array<float, 4> nms_normalize_box(const float* b)
{
    if constexpr(CenterPointBox)
    {
        const float xc = b[0];
        const float yc = b[1];
        const float hw = b[2] * 0.5f;
        const float hh = b[3] * 0.5f;
        return {xc - hw, yc - hh, xc + hw, yc + hh};
    }
    else
    {
        // ONNX layout: [y1, x1, y2, x2]; corners may be in either order.
        const float y1   = b[0];
        const float x1   = b[1];
        const float y2   = b[2];
        const float x2   = b[3];
        const float xmin = min(x1, x2);
        const float xmax = max(x1, x2);
        const float ymin = min(y1, y2);
        const float ymax = max(y1, y2);
        return {xmin, ymin, xmax, ymax};
    }
}

__device__ inline bool
nms_iou_over_threshold(const array<float, 4>& a, const array<float, 4>& b, float threshold)
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

struct nms_score_greater
{
    constexpr bool operator()(const nms_data& a, const nms_data& b) const
    {
        return a.score > b.score;
    }
};

// Phase 1: load (score, box, box_index) tuples into a per-block buffer of
// AlignedN entries (power of two), padding the [N, AlignedN) tail with sentinel
// values, then sort the buffer in descending order by score.
template <bool CenterPointBox, index_int N, index_int AlignedN>
__device__ void nms_load_and_sort(index idx,
                                  const float* boxes_b,   // [N, 4]
                                  const float* scores_bc, // [N]
                                  nms_data* sorted)
{
    idx.local_stride(AlignedN, [&](auto i) {
        nms_data d;
        if(i < N)
        {
            d.score     = scores_bc[i];
            d.box       = nms_normalize_box<CenterPointBox>(boxes_b + i * 4);
            d.box_index = static_cast<int>(i);
        }
        else
        {
            // Sentinel: -inf score so it never beats any real entry, and a
            // negative box_index so accidental dereferencing is detectable.
            d.score     = -__FLT_MAX__;
            d.box       = array<float, 4>{0.f, 0.f, 0.f, 0.f};
            d.box_index = -1;
        }
        sorted[i] = d;
    });
    __syncthreads();
    bitonic_sort<nms_score_greater>{nms_score_greater{}}.template block_sort<AlignedN>(idx, sorted);
}

// Phase 2: build the packed upper-triangular IoU mask for the N sorted boxes.
// Work is striped (i, N-1-i) per thread so each thread does roughly the same
// amount of work regardless of where it falls in the triangle.
template <index_int N>
__device__ void nms_make_iou_mask(index idx, const nms_data* sorted, uint8_t* mask, float iou_thr)
{
    constexpr index_int half = N / 2;

    auto fill_row = [&](index_int i) {
        for(index_int j = i + 1; j < N; ++j)
        {
            mask[nms_packed_idx(i, j, N)] =
                nms_iou_over_threshold(sorted[i].box, sorted[j].box, iou_thr) ? 1 : 0;
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

// Phase 3: greedy filter, mirroring the prototype but using a global atomic
// counter to compact outputs from all (batch, class) blocks into a single
// dense output buffer.
template <index_int N>
__device__ void nms_filter_atomic(index idx,
                                  const nms_data* sorted,
                                  const uint8_t* mask,
                                  int batch_idx,
                                  int class_idx,
                                  index_int max_output,
                                  float score_thr,
                                  unsigned long long* global_count, // NOLINT
                                  int64_t* output,
                                  index_int output_capacity)
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
                const unsigned long long slot = atomicAdd(global_count, 1ull); // NOLINT
                if(slot < static_cast<unsigned long long>(output_capacity))
                {
                    output[slot * 3 + 0] = batch_idx;
                    output[slot * 3 + 1] = class_idx;
                    output[slot * 3 + 2] = sorted[i].box_index;
                }
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
}

// Per-block driver: one block per (batch_idx, class_idx). Workspace pointers
// are sliced into per-block segments using idx.group.
template <bool CenterPointBox,
          index_int NumBatches,
          index_int NumClasses,
          index_int NumBoxes,
          index_int AlignedNumBoxes,
          class Boxes,
          class Scores,
          class MaxOut,
          class IouThr,
          class ScoreThr,
          class Sorted,
          class Mask,
          class Count,
          class Output>
__device__ void nonmaxsuppression(Boxes boxes,
                                  Scores scores,
                                  MaxOut max_out_p,
                                  IouThr iou_thr_p,
                                  ScoreThr score_thr_p,
                                  Sorted sorted_buf,
                                  Mask mask_buf,
                                  Count count_buf,
                                  Output output)
{
    static_assert(NumBatches > 0, "num_batches must be > 0");
    static_assert(NumClasses > 0, "num_classes must be > 0");

    auto idx                            = make_index();
    const index_int block_id            = idx.group;
    const int batch_idx                 = static_cast<int>(block_id / NumClasses);
    const int class_idx                 = static_cast<int>(block_id % NumClasses);
    constexpr index_int iou_packed_size = (NumBoxes > 1) ? (NumBoxes * (NumBoxes - 1)) / 2 : 1;

    nms_data* my_sorted =
        reinterpret_cast<nms_data*>(sorted_buf.data()) + block_id * AlignedNumBoxes;
    uint8_t* my_mask = reinterpret_cast<uint8_t*>(mask_buf.data()) + block_id * iou_packed_size;

    const float* boxes_b   = boxes.data() + batch_idx * NumBoxes * 4;
    const float* scores_bc = scores.data() + (batch_idx * NumClasses + class_idx) * NumBoxes;

    // Pull scalar tensor inputs once. They're broadcast to all threads via the
    // common load (each thread reads the same single element).
    const int64_t max_out_val = max_out_p[0];
    const float iou_thr_val   = iou_thr_p[0];
    const float score_thr_val = score_thr_p[0];

    nms_load_and_sort<CenterPointBox, NumBoxes, AlignedNumBoxes>(
        idx, boxes_b, scores_bc, my_sorted);
    __syncthreads();

    if constexpr(NumBoxes > 1)
    {
        nms_make_iou_mask<NumBoxes>(idx, my_sorted, my_mask, iou_thr_val);
        __syncthreads();
    }

    // The CPU op reads max_output_boxes_per_class as std::size_t, so a negative
    // signed value is treated as a very large unsigned (effectively unlimited).
    // Mirror that here by reinterpreting as unsigned and then capping at
    // NumBoxes, which is the most we could ever emit per (batch, class) block.
    const auto max_unsigned         = static_cast<uint64_t>(max_out_val);
    const index_int max_output      = (max_unsigned > static_cast<uint64_t>(NumBoxes))
                                          ? static_cast<index_int>(NumBoxes)
                                          : static_cast<index_int>(max_unsigned);
    const index_int output_capacity = output.get_shape().lens[0];
    auto* count_addr =
        reinterpret_cast<unsigned long long*>(count_buf.data()); // NOLINT
    nms_filter_atomic<NumBoxes>(idx,
                                my_sorted,
                                my_mask,
                                batch_idx,
                                class_idx,
                                max_output,
                                score_thr_val,
                                count_addr,
                                output.data(),
                                output_capacity);
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_NONMAXSUPPRESSION_HPP
