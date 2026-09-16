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
#ifndef MIGRAPHX_GUARD_KERNELS_INT4_GEMV_HPP
#define MIGRAPHX_GUARD_KERNELS_INT4_GEMV_HPP

#include <migraphx/kernels/hip.hpp>
#include <migraphx/kernels/types.hpp>

namespace migraphx {

using half2_t = _Float16 __attribute__((ext_vector_type(2)));

// Packed fp16 dot-product with fp32 accumulate.
//
// v_dot2_f32_f16 does not exist on every GPU target, and the builtin that emits it is
// not declared on targets that lack it -- so this must be guarded at preprocessing
// time, not left to instruction selection.  Where the builtin is unavailable the
// scalar form below is the arithmetic it replaced.  The accumulator is fp32 in both
// paths, but the fallback is two separate adds where the builtin fuses, so results may
// differ in the last ulp -- it is the same computation, not a bit-identical one.
#ifndef MIGRAPHX_INT4_GEMV_HAS_FDOT2
#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_fdot2)
#define MIGRAPHX_INT4_GEMV_HAS_FDOT2 1
#else
#define MIGRAPHX_INT4_GEMV_HAS_FDOT2 0
#endif
#else
#define MIGRAPHX_INT4_GEMV_HAS_FDOT2 0
#endif
#endif

__device__ inline float int4_gemv_dot2(half2_t a, half2_t b, float acc)
{
#if MIGRAPHX_INT4_GEMV_HAS_FDOT2
    return __builtin_amdgcn_fdot2(a, b, acc, false);
#else
    return acc + static_cast<float>(a[0]) * static_cast<float>(b[0]) +
           static_cast<float>(a[1]) * static_cast<float>(b[1]);
#endif
}

// Branch-free nibble -> fp16 conversion by exponent injection.
//
// A half whose bits are (0x4C00 | (n << 6)) is exactly 16 + n for n in [0,16): 0x4C00 is
// 16.0, and in the binade [16,32) the fp16 ulp is 16 * 2^-10 = 1/64, so placing n at
// mantissa bits 6..9 adds exactly n.  No conversion instruction, no rounding, no table.
//
// The bias is what makes it free, and it is also the only cost.  The dot product then
// computes sum(a_i * (16 + n_i)) = raw_dot + 16 * a_sum, and a_sum is already computed for
// the zero-point term, so the bias is removed by folding 16 into that same subtraction --
// zero extra work in the inner loop.
//
// 16 is chosen over the more common 1024 (0x6400, the first binade with ulp 1) on
// precision grounds.  The subtraction is a cancellation: the quantity carried through the
// accumulator is ~bias * a_sum while the answer is ~8 * a_sum, so the fraction of the fp32
// mantissa left for the answer falls as the bias rises.  1024 throws away ~7 bits; 16
// throws away ~1.  Both need the nibble to land on an integer ulp, which is what pins the
// choice to a power of two >= 16.
#define INT4_GEMV_NIB_BIAS 16.0f
#define INT4_GEMV_NIB_BASE 0x4C004C00u
#define INT4_GEMV_NIB_MASK 0x03C003C0u

// INT4 M=1 GEMV with factored dequantization.
//
// Template params:
//   BLOCK_SIZE: threads per workgroup (K-parallel workers)
//   TILE_N:     output columns per workgroup-row
//   BLOCK_K:    elements per quantization group (block_size in ONNX parlance)
//   HAS_ZP:     true for asymmetric quantization (zero_points present)
//
// Inputs:
//   a_ptr:      activation [batch, 1, K], fp16, row-major
//   b_ptr:      packed INT4 weights [N, K/2], uint8 (2 nibbles per byte, low nibble first)
//   scales_ptr: per-group scale [N, k_blocks], fp16, compact (k_blocks = K/BLOCK_K)
//   zp_ptr:     per-group zero-point [N, k_blocks] uint8, or [N, k_blocks/2]
//               nibble-packed; or nullptr
//   out_ptr:    output [batch, 1, N], fp16
//
// Grid: flat 1D, global = ceil(N/TILE_N) * batch * BLOCK_SIZE
// Block: (BLOCK_SIZE, 1, 1)
// N_BLOCKS = ceil(N/TILE_N); batch_id = blockIdx.x / N_BLOCKS; n_block = blockIdx.x % N_BLOCKS

template <int BLOCK_SIZE,
          int TILE_N,
          int BLOCK_K,
          bool HAS_ZP,
          uint32_t N_BLOCKS,
          bool HAS_BIAS = false>
__device__ void int4_gemv_kernel(const _Float16* __restrict__ a_ptr,
                                 const uint8_t* __restrict__ b_ptr,
                                 const _Float16* __restrict__ scales_ptr,
                                 const uint8_t* __restrict__ zp_ptr,
                                 const _Float16* __restrict__ bias_ptr,
                                 _Float16* __restrict__ out_ptr,
                                 uint32_t N,
                                 uint32_t K)
{
    const uint32_t tid      = threadIdx.x;
    const uint32_t flat_bid = blockIdx.x;
    const uint32_t n_block  = flat_bid % N_BLOCKS;
    const uint32_t batch_id = flat_bid / N_BLOCKS;

    const uint32_t k_per_iter = 32; // process 32 K elements per inner iteration (128-bit B load)

    constexpr int WARP_SIZE = MIGRAPHX_WAVEFRONTSIZE;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    const int warp_id       = tid / WARP_SIZE;
    const int lane_id       = tid % WARP_SIZE;

    const _Float16* a_row = a_ptr + static_cast<uint64_t>(batch_id) * K;
    _Float16* out_row     = out_ptr + static_cast<uint64_t>(batch_id) * N;

    float partial[TILE_N];
    for(int tn = 0; tn < TILE_N; tn++)
        partial[tn] = 0.0f;

    // Each thread strides over K dimension
    for(uint32_t k_start = tid * k_per_iter; k_start < K; k_start += BLOCK_SIZE * k_per_iter)
    {
        // Load 32 activation values into registers, compute a_sum.
        //
        // Held as _Float16, not float: this array is a fixed 32-VGPR tax paid by
        // every config, on top of the TILE_N-scaled b_pk/partial/sv, and it is what
        // drags occupancy down as TILE_N rises (at TN=16 it is 32 of ~130 VGPRs).
        // Keeping the loaded fp16 bit pattern and widening at each use is
        // numerically identical -- the value was already fp16 in memory, and the
        // accumulator stays fp32 -- but halves the cost to 16 VGPRs.
        _Float16 a_vals[32];
        float a_sum          = 0.0f;
        const uint32_t k_end = (k_start + k_per_iter < K) ? k_per_iter : (K - k_start);

#pragma unroll
        for(uint32_t i = 0; i < 32; i++)
        {
            _Float16 v = (i < k_end) ? a_row[k_start + i] : static_cast<_Float16>(0.0f);
            a_vals[i]  = v;
            a_sum += static_cast<float>(v);
        }

        // Activations repacked as half2 pairs (i, i+4) within each 8-nibble group.
        //
        // The pairing is (i, i+4) rather than (i, i+1) because that is the pairing the
        // weight side can produce for free: one mask `w & 0x000F000F` lifts nibble j and
        // nibble j+4 of a 32-bit word into the two halves of a packed register in a single
        // op.  Pairing (i, i+1) would need a shift and a merge per pair.  The dot product
        // does not care about order, so the cheap pairing is the right one -- but the two
        // sides must agree, which is the only reason this loop looks the way it does.
        //
        // Built once per K-iteration and reused across all TILE_N columns, so its cost is
        // amortised by TILE_N while the weight-side unpack it enables is paid per column.
        half2_t a_pk[16];
#pragma unroll
        for(int g = 0; g < 4; g++)
#pragma unroll
            for(int j = 0; j < 4; j++)
            {
                a_pk[g * 4 + j][0] = a_vals[g * 8 + j];
                a_pk[g * 4 + j][1] = a_vals[g * 8 + j + 4];
            }

        // Precompute symmetric ZP contribution once per K-iteration.
        //
        // ZP=8 for symmetric INT4, plus INT4_GEMV_NIB_BIAS because the unpack produces
        // (bias + n), not n -- see the comment on the constant.  The two corrections are
        // the same shape (a scalar multiple of a_sum) so folding them costs nothing.
        const float a_sum_x_zp    = HAS_ZP ? 0.0f : (a_sum * (8.0f + INT4_GEMV_NIB_BIAS));
        const uint32_t grp        = k_start / BLOCK_K;
        const uint32_t n_k_blocks = K / BLOCK_K;

        // Load phase: fetch all B vectors + scales for this tile before computing any dots
        uint4 b_pk[TILE_N];
        float sv[TILE_N];
#pragma unroll
        for(int tn = 0; tn < TILE_N; tn++)
        {
            uint32_t n_idx = n_block * TILE_N + tn;
            if(n_idx < N)
            {
                // 128-bit vector load: 16 bytes = 32 packed nibbles
                // B layout: [N, K/2], row-major
                const uint8_t* b_col = b_ptr + static_cast<uint64_t>(n_idx) * (K / 2) + k_start / 2;

                if(k_end == 32)
                {
                    // Non-temporal: each weight byte is read exactly once per decode step and
                    // never revisited, so caching it only evicts the activation and scale lines
                    // that ARE reused.
                    //
                    // Goes through a native ext_vector rather than uint4: HIP's uint4 is a
                    // HIP_vector_type class, which the nontemporal builtin rejects.  Keeping it
                    // a 4-wide vector is what holds this to a single 128-bit dwordx4 -- four
                    // scalar loads would trade a cache hint for three extra memory ops.
                    using u32x4 = unsigned int __attribute__((ext_vector_type(4)));
                    u32x4 w     = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(b_col));
                    b_pk[tn].x  = w.x;
                    b_pk[tn].y  = w.y;
                    b_pk[tn].z  = w.z;
                    b_pk[tn].w  = w.w;
                }
                else
                {
                    b_pk[tn]           = {0, 0, 0, 0};
                    const uint8_t* src = b_col;
                    uint8_t* dst       = reinterpret_cast<uint8_t*>(&b_pk[tn]);
                    for(uint32_t i = 0; i < k_end / 2; i++)
                        dst[i] = src[i];
                }

                sv[tn] =
                    static_cast<float>(scales_ptr[static_cast<uint64_t>(n_idx) * n_k_blocks + grp]);
            }
        }

        // Compute phase: dot products for each output column in this tile
#pragma unroll
        for(int tn = 0; tn < TILE_N; tn++)
        {
            uint32_t n_idx = n_block * TILE_N + tn;
            if(n_idx >= N)
                break;

            // Compute the biased dot product sum(a[i] * (BIAS + w_raw[i])).
            //
            // The packed form replaces 32 scalar FMAs and 32 int->float conversions per
            // column with 16 packed fp16 dots and 12 integer ops.  Each `v_dot2_f32_f16`
            // consumes a half2 of activations and a half2 of weights and accumulates into fp32, so
            // the accumulator stays fp32 exactly as before -- only the multiply operands
            // are narrowed, and they were already fp16 in memory.
            //
            // On a target without v_dot2_f32_f16, int4_gemv_dot2 falls back to two scalar
            // FMAs -- the code this replaced.  The unpack saving survives either way.
            float raw_dot           = 0.0f;
            const uint32_t words[4] = {b_pk[tn].x, b_pk[tn].y, b_pk[tn].z, b_pk[tn].w};
#pragma unroll
            for(int g = 0; g < 4; g++)
            {
#pragma unroll
                for(int j = 0; j < 4; j++)
                {
                    // Lifts nibble j and nibble j+4 of the word into the two halves at
                    // once -- one shift, one mask, one or, for two weights.
                    uint32_t p =
                        INT4_GEMV_NIB_BASE | (((words[g] >> (4 * j)) << 6) & INT4_GEMV_NIB_MASK);
                    half2_t bv;
                    __builtin_memcpy(&bv, &p, sizeof(p));
                    raw_dot = int4_gemv_dot2(a_pk[g * 4 + j], bv, raw_dot);
                }
            }

            // Factored dequant: result = (raw_dot - a_sum * (zp + BIAS)) * scale.
            // BIAS is folded into the zero-point subtraction, so removing it is free.
            if constexpr(HAS_ZP)
            {
                float zv =
                    static_cast<float>(zp_ptr[static_cast<uint64_t>(n_idx) * n_k_blocks + grp]);
                partial[tn] += (raw_dot - a_sum * (zv + INT4_GEMV_NIB_BIAS)) * sv[tn];
            }
            else
            {
                partial[tn] += (raw_dot - a_sum_x_zp) * sv[tn];
            }
        }
    }

    // Warp-shuffle reduction (zero LDS for the single-warp case)
#pragma unroll
    for(int tn = 0; tn < TILE_N; tn++)
        for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            partial[tn] += __shfl_down(partial[tn], offset, WARP_SIZE);

    if constexpr(NUM_WARPS == 1)
    {
        // Single warp: shuffle has the final result, zero shared memory
        if(lane_id == 0)
        {
            for(int tn = 0; tn < TILE_N; tn++)
            {
                uint32_t n_idx = n_block * TILE_N + tn;
                if(n_idx < N)
                {
                    float val = partial[tn];
                    if constexpr(HAS_BIAS)
                        val += static_cast<float>(bias_ptr[n_idx]);
                    out_row[n_idx] = static_cast<_Float16>(val);
                }
            }
        }
    }
    else
    {
        // Multi-warp: each warp writes its partial to minimal shared memory,
        // then warp 0 does a final shuffle reduction
        __shared__ float warp_sums[TILE_N * NUM_WARPS];

        if(lane_id == 0)
        {
            for(int tn = 0; tn < TILE_N; tn++)
                warp_sums[tn * NUM_WARPS + warp_id] = partial[tn];
        }
        __syncthreads();

        // Warp 0 reduces across warps
        if(warp_id == 0)
        {
            for(int tn = 0; tn < TILE_N; tn++)
            {
                float val = (lane_id < NUM_WARPS) ? warp_sums[tn * NUM_WARPS + lane_id] : 0.0f;
                for(int offset = NUM_WARPS / 2; offset > 0; offset >>= 1)
                    val += __shfl_down(val, offset, WARP_SIZE);

                if(lane_id == 0)
                {
                    uint32_t n_idx = n_block * TILE_N + tn;
                    if(n_idx < N)
                    {
                        if constexpr(HAS_BIAS)
                            val += static_cast<float>(bias_ptr[n_idx]);
                        out_row[n_idx] = static_cast<_Float16>(val);
                    }
                }
            }
        }
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_INT4_GEMV_HPP
