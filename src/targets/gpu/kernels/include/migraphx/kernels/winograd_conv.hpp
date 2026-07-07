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
#ifndef MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP
#define MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

// gfx12 buffer-resource word 3 constant (from composable_kernel).
// Setting this in the SRD makes raw_buffer_load_* return 0 for OOB accesses,
// which lets us drop the per-element bounds checks in the input transform.
constexpr uint32_t buffer_rsrc_3rd_dword_gfx12 = 0x31004000;

// Quad of WMMAs in a single inline-asm block. Forces the compiler to issue
// them back-to-back (each is 8-cycle wait state but to a DIFFERENT
// accumulator, so the next can issue ~1 cycle later). The compiler is then
// free to schedule the alpha-fold v_add_f32 ops *outside* this block, which
// gives them a continuous block of VALU cycles to fill while the matrix pipe
// processes the WMMAs.
// NOLINTNEXTLINE(readability-function-size): fixed 4-way WMMA unroll, one asm block
__device__ inline void wmma_quad_asm(vec<half, 8> a0,
                                     vec<half, 8> b0,
                                     vec<half, 8> a1,
                                     vec<half, 8> b1,
                                     vec<half, 8> a2,
                                     vec<half, 8> b2,
                                     vec<half, 8> a3,
                                     vec<half, 8> b3,
                                     vec<float, 8>& m0,
                                     vec<float, 8>& m1,
                                     vec<float, 8>& m2,
                                     vec<float, 8>& m3)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
                 "v_wmma_f32_16x16x16_f16 %1, %6, %7, %1\n\t"
                 "v_wmma_f32_16x16x16_f16 %2, %8, %9, %2\n\t"
                 "v_wmma_f32_16x16x16_f16 %3, %10, %11, %3"
                 : "+v"(m0), "+v"(m1), "+v"(m2), "+v"(m3)
                 : "v"(a0), "v"(b0), "v"(a1), "v"(b1), "v"(a2), "v"(b2), "v"(a3), "v"(b3));
}

// Octet of WMMAs in a single inline-asm block. Costs 8 live fp32 vec<8>
// accumulators (64 VGPRs) but gives the matrix pipe a continuous run.
// NOLINTNEXTLINE(readability-function-size): fixed 8-way WMMA unroll, one asm block
__device__ inline void wmma_octet_asm(vec<half, 8> a0,
                                      vec<half, 8> b0,
                                      vec<half, 8> a1,
                                      vec<half, 8> b1,
                                      vec<half, 8> a2,
                                      vec<half, 8> b2,
                                      vec<half, 8> a3,
                                      vec<half, 8> b3,
                                      vec<half, 8> a4,
                                      vec<half, 8> b4,
                                      vec<half, 8> a5,
                                      vec<half, 8> b5,
                                      vec<half, 8> a6,
                                      vec<half, 8> b6,
                                      vec<half, 8> a7,
                                      vec<half, 8> b7,
                                      vec<float, 8>& m0,
                                      vec<float, 8>& m1,
                                      vec<float, 8>& m2,
                                      vec<float, 8>& m3,
                                      vec<float, 8>& m4,
                                      vec<float, 8>& m5,
                                      vec<float, 8>& m6,
                                      vec<float, 8>& m7)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %8, %9, %0\n\t"
                 "v_wmma_f32_16x16x16_f16 %1, %10, %11, %1\n\t"
                 "v_wmma_f32_16x16x16_f16 %2, %12, %13, %2\n\t"
                 "v_wmma_f32_16x16x16_f16 %3, %14, %15, %3\n\t"
                 "v_wmma_f32_16x16x16_f16 %4, %16, %17, %4\n\t"
                 "v_wmma_f32_16x16x16_f16 %5, %18, %19, %5\n\t"
                 "v_wmma_f32_16x16x16_f16 %6, %20, %21, %6\n\t"
                 "v_wmma_f32_16x16x16_f16 %7, %22, %23, %7"
                 : "+v"(m0), "+v"(m1), "+v"(m2), "+v"(m3), "+v"(m4), "+v"(m5), "+v"(m6), "+v"(m7)
                 : "v"(a0),
                   "v"(b0),
                   "v"(a1),
                   "v"(b1),
                   "v"(a2),
                   "v"(b2),
                   "v"(a3),
                   "v"(b3),
                   "v"(a4),
                   "v"(b4),
                   "v"(a5),
                   "v"(b5),
                   "v"(a6),
                   "v"(b6),
                   "v"(a7),
                   "v"(b7));
}

__device__ inline auto make_input_buffer_rsrc(const half* p, uint32_t byte_count)
{
    // The builtin takes a non-const base pointer, but the input tensor is const.
    auto* base = const_cast<half*>(p); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    return __builtin_amdgcn_make_buffer_rsrc(base, 0, byte_count, buffer_rsrc_3rd_dword_gfx12);
}

// Lane-indexed raw buffer load of a single fp16. OOB returns 0.
__device__ inline half buffer_load_half(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    uint16_t v = __builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0);
    return bit_cast<half>(v);
}

// Lane-indexed raw buffer load of 4 fp16 = 8 bytes. OOB bytes return 0.
// Caller is responsible for alignment (byte_offset divisible by 4 to avoid
// faulting; gfx12 buffer loads tolerate 4-byte alignment for b64).
__device__ inline vec<half, 4> buffer_load_half4(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    auto v = __builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0);
    return bit_cast<vec<half, 4>>(v);
}

// Lane-indexed raw buffer load of 8 fp16 = 16 bytes (b128). OOB bytes return 0.
// Used by the NHWC input load, where 8 contiguous channels are read at once.
__device__ inline vec<half, 8> buffer_load_half8(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    auto v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
    return bit_cast<vec<half, 8>>(v);
}

// F(2x2, 3x3) Winograd transforms used inline by the WMMA path.
// B^T (input):  | 1  0 -1  0 |     A^T (output): | 1  1  1  0 |
//               | 0  1  1  0 |                   | 0  1 -1 -1 |
//               | 0 -1  1  0 |
//               | 0  1  0 -1 |

__device__ inline array<half, 16> winograd_input_transform_f23(const array<half, 16>& d_arr)
{
    using h2 = vec<half, 2>;
    // First pass B^T d: per-row sub/add of 4 columns. All 4 columns are
    // independent so we express each row as TWO packed half2 ops --
    // generates v_pk_add_f16 / v_pk_sub_f16.
    const h2* dp   = as_vec<2>(d_arr.data());
    const h2 d0_lo = dp[0]; // d[0,0..1]
    const h2 d0_hi = dp[1]; // d[0,2..3]
    const h2 d1_lo = dp[2];
    const h2 d1_hi = dp[3];
    const h2 d2_lo = dp[4];
    const h2 d2_hi = dp[5];
    const h2 d3_lo = dp[6];
    const h2 d3_hi = dp[7];

    const h2 t0_lo = d0_lo - d2_lo;
    const h2 t0_hi = d0_hi - d2_hi;
    const h2 t1_lo = d1_lo + d2_lo;
    const h2 t1_hi = d1_hi + d2_hi;
    const h2 t2_lo = d2_lo - d1_lo;
    const h2 t2_hi = d2_hi - d1_hi;
    const h2 t3_lo = d1_lo - d3_lo;
    const h2 t3_hi = d1_hi - d3_hi;

    // Second pass d B: V[i,j] in terms of t[i,*].
    //   V[i,0] = t[i,0] - t[i,2]
    //   V[i,1] = t[i,1] + t[i,2]
    //   V[i,2] = t[i,2] - t[i,1]
    //   V[i,3] = t[i,1] - t[i,3]
    array<half, 16> v;
    auto fill = [&](int row, h2 t_lo, h2 t_hi) {
        v[row * 4 + 0] = t_lo.x - t_hi.x;
        v[row * 4 + 1] = t_lo.y + t_hi.x;
        v[row * 4 + 2] = t_hi.x - t_lo.y;
        v[row * 4 + 3] = t_lo.y - t_hi.y;
    };
    fill(0, t0_lo, t0_hi);
    fill(1, t1_lo, t1_hi);
    fill(2, t2_lo, t2_hi);
    fill(3, t3_lo, t3_hi);
    return v;
}

// Element-generic B^T d B input transform: same math as above but over an
// arbitrary element type T. The NHWC load path uses T = vec<half, 8> so the
// transform runs across the lane's 8 contiguous channels at once (the loads
// already deliver them packed), avoiding the per-channel scalar transpose. Only
// per-element indexing is used (never array's whole-array vectorized ops), so a
// vec element type is fine here.
template <class T>
__device__ inline void winograd_input_transform_f23_vec(const array<T, 16>& d, array<T, 16>& v)
{
    array<T, 16> t;
    repeat_c<4>([&](auto j) {
        t[0 * 4 + j] = d[0 * 4 + j] - d[2 * 4 + j];
        t[1 * 4 + j] = d[1 * 4 + j] + d[2 * 4 + j];
        t[2 * 4 + j] = d[2 * 4 + j] - d[1 * 4 + j];
        t[3 * 4 + j] = d[1 * 4 + j] - d[3 * 4 + j];
    });
    repeat_c<4>([&](auto i) {
        v[i * 4 + 0] = t[i * 4 + 0] - t[i * 4 + 2];
        v[i * 4 + 1] = t[i * 4 + 1] + t[i * 4 + 2];
        v[i * 4 + 2] = t[i * 4 + 2] - t[i * 4 + 1];
        v[i * 4 + 3] = t[i * 4 + 1] - t[i * 4 + 3];
    });
}

// WMMA-based winograd F(2x2, 3x3) kernel for gfx12 wave32.
//
// Per WMMA call (16x16x16 fp16 -> fp32):
//   A operand: lane l holds A[m=l%16, k=(l/16)*8 + i] for i in 0..7  (V8h)
//   B operand: lane l holds B[k=l%16, n=(l/16)*8 + i] for i in 0..7  (V8h)
//   D output : lane l holds D[m=(l/16)*8 + i, n=l%16] for i in 0..7  (V8f)
//
// Block organization:
//   - workgroup = NW waves * 32 lanes (wave32)
//   - All NW waves share K_block (bk = 16 = WMMA M dim); waves split NT
//   - Per wave: 16 K x 16 NT outputs per wp = 256 / 32 lanes = 8 per lane
//   - We stream the output transform per WMMA so we don't keep all 16 wp
//     accumulators alive simultaneously (which would force register spill).
//   - CB must be a multiple of 16 (WMMA K dim).

// PostInput is the type the conv result is cast to before the fused
// pointwise `f` runs. It is the conv's natural output type (half) so the
// post-op computes at conv precision and matches the reference — UNLESS the
// pointwise's first op converts the result to a wider type, in which case the
// codegen passes that wider type here so we feed `f` the fp32 accumulator
// directly instead of round-tripping it through half. The final store still
// uses the actual output type (`f` converts to it).
template <index_int NW,
          index_int CB,
          index_int KW,
          index_int SK,
          bool FT,
          bool NHWC,
          class PostInput,
          class F,
          class Output,
          class Input,
          class Weights,
          class... Inputs>
// NOLINTNEXTLINE(readability-function-size): single fused winograd+WMMA+writeback kernel
__device__ void winograd_conv_f23_wmma(F f, Output output, Input x, Weights u, Inputs... inputs)
{
    static_assert(CB % 16 == 0, "CB must be a multiple of WMMA K (16)");
    static_assert(KW >= 1, "KW must be >= 1");
    static_assert(SK >= 1 and SK <= NW and (NW % SK) == 0, "SK must divide NW evenly");
    // SK = within-WG c-axis split factor. SK waves cooperate to reduce the
    // C contraction; NW/SK independent NT-groups exist per workgroup so
    // bt = bt_per_wave * (NW/SK). SK=1 is the original (no split) path.
    // For SK>1, KW must be 1 (LDS budget would otherwise overflow with the
    // per-wave U_lds slots).
    static_assert(SK == 1 or KW == 1, "SK>1 currently requires KW==1");
    constexpr index_int bk          = 16;
    constexpr index_int bt_per_wave = 16;
    constexpr index_int nt_groups   = NW / SK;
    constexpr index_int bt          = bt_per_wave * nt_groups;
    constexpr index_int bk_wg       = bk * KW;

    auto idx = make_index();

    auto out_shape = output.get_shape();
    auto x_shape   = x.get_shape();

    const auto n     = out_shape.lens[0];
    const auto out_c = out_shape.lens[1];
    const auto out_h = out_shape.lens[2];
    const auto out_w = out_shape.lens[3];
    const auto in_c  = x_shape.lens[1];
    const auto in_h  = x_shape.lens[2];
    const auto in_w  = x_shape.lens[3];

    const auto tiles_w       = (out_w + 1) / 2;
    const auto tiles_h       = (out_h + 1) / 2;
    const auto tiles_per_img = tiles_h * tiles_w;
    const auto nt_total      = n * tiles_per_img;

    // Each workgroup covers KW consecutive K_blocks for one t_block. The V
    // transform is shared across all KW k_blocks, so we only pay the
    // V-transform cost once per (t_block, c_block) instead of KW times.
    const auto k_wg_blocks = (out_c + bk_wg - 1) / bk_wg;
    const auto k_wg_block  = idx.group % k_wg_blocks;
    const auto t_block     = idx.group / k_wg_blocks;
    const auto k_base      = k_wg_block * bk_wg;
    const auto t_base      = t_block * bt;

    const auto lane    = idx.local % 32;
    const auto wave_id = idx.local / 32;

    // ---- Split-c: wave_id is split into NT-group + sk-part ----
    // For SK=1: nt_groups=NW, wave_sk_part is always 0, wave_nt_idx == wave_id.
    // For SK>1: NW/SK NT-groups; SK waves per group cooperate on c contraction.
    const index_int wave_nt_idx  = wave_id / SK;
    const index_int wave_sk_part = wave_id % SK;

    // ---- V layout (REGISTER-RESIDENT) ----
    // V values are kept in per-lane registers instead of LDS. The lane
    // assignment is chosen so that each lane already holds the exact 8 fp16
    // values that the WMMA B operand expects.
    //   nt_lane = (wave_nt_idx*bt_per_wave) + lane%16  (1 nt per lane)
    //   c_lane in c_off + 0..7  with c_off = (lane/16)*8   (8 c per lane)
    static_assert(CB == 16 or CB == 32, "DPP V path supports CB=16 or CB=32");
    constexpr index_int v_chunks = CB / 16;
    constexpr index_int wp_count = 16;
    // U weight staging: for SK==1 every wave of the WG reads the SAME U
    // (same K block) at every cb iter, so the cooperative-into-LDS pattern
    // pays for itself (one shared load served by NW lanes' worth of read
    // bandwidth from LDS). For SK>1 each wave reads its OWN c slice, so the
    // LDS slots are per-wave and there is no cross-wave sharing — in that
    // case we skip LDS entirely and stream U directly from global into the
    // WMMA A operand (the buffer cache handles reuse within the wave).
    constexpr bool u_via_lds       = (SK == 1);
    constexpr index_int u_slots    = u_via_lds ? KW : 1;
    constexpr index_int u_slot_len = 16 * bk * CB;
    constexpr index_int u_smem_len = u_via_lds ? u_slots * u_slot_len : 1;
    __shared__ uninitialized_buffer<half, u_smem_len> u_smem;
    // y_reduce_lds: holds per-wave y_partial during the SK>1 cross-wave
    // reduce. nt_groups groups × SK waves × 32 lanes × 32 fp32 (KW=1 only).
    // Sized to 1 element for SK=1 to keep the LDS allocation valid but tiny.
    constexpr index_int y_red_len = (SK > 1) ? (nt_groups * SK * 32 * 32) : 1;
    __shared__ uninitialized_buffer<float, y_red_len> y_reduce_lds;

    auto u_cache_idx = [&](index_int slot, index_int wp, index_int k, index_int c) {
        return slot * u_slot_len + wp * bk * CB + k * CB + c;
    };

    // alpha[wp,r,c] = A^T[r, wp/4] * A[wp%4, c]
    constexpr array<array<float, 4>, 2> at = {array<float, 4>{1.f, 1.f, 1.f, 0.f},
                                              array<float, 4>{0.f, 1.f, -1.f, -1.f}};

    // Y[k_idx][r*2+c] running accumulator, one vec<float,8> per output
    // position (8 K rows per lane). KW * 4 outputs. Using vec (the native
    // WMMA accumulator type) instead of array lets the output transform fold
    // run as packed vec adds rather than per-lane scalar extraction.
    array<array<vec<float, 8>, 4>, KW> y{};

    // Buffer descriptors for X (input) and U (weights). raw_buffer_load
    // returns 0 for OOB byte offsets so we don't need explicit bounds checks.
    const auto x_sh             = x_shape.strides;
    const auto* x_data          = x.data();
    const uint32_t x_byte_count = static_cast<uint32_t>(x_shape.element_space()) * sizeof(half);
    auto x_rsrc                 = make_input_buffer_rsrc(x_data, x_byte_count);

    const auto* u_data = u.data();
    const uint32_t u_byte_count =
        static_cast<uint32_t>(u.get_shape().element_space()) * sizeof(half);
    auto u_rsrc = make_input_buffer_rsrc(u_data, u_byte_count);
    // U layout: [4 or 3, 3, K, C] -- strides for byte offset computation.
    const auto u_sh = u.get_shape().strides;

    const auto cblocks = (in_c + CB - 1) / CB;

    // V-in-registers storage: v_lane[c_chunk][c_in_chunk][wp].
    // For CB=16: v_chunks=1, c_in_chunk in 0..7 (lane's 8 c values).
    // For CB=32: v_chunks=2 (two c-chunks; one b128 read per chunk for WMMA).
    array<array<array<half, wp_count>, 8>, v_chunks> v_lane;

    // Cached per-wave / per-lane geometry. With SK>1 each NT-group occupies
    // SK consecutive waves that all map to the SAME nt range, so we use
    // wave_nt_idx (not raw wave_id) to compute the NT base.
    const index_int wave_nt_base_in_block = wave_nt_idx * bt_per_wave;
    const index_int m_in_wave             = lane % 16;
    const index_int c_off                 = (lane / 16) * 8;
    const index_int nt_slot               = wave_nt_base_in_block + m_in_wave;
    const index_int nt_global             = t_base + nt_slot;

    // Lane's tile (n, th, tw). Same for all c (only c changes per tile_idx).
    // Decompose the linear nt index over the {n, tiles_h, tiles_w} extents.
    const bool nt_active   = (nt_global < nt_total);
    const auto tile_idx    = nt_active ? array<index_int, 3>{n, tiles_h, tiles_w}.multi(nt_global)
                                       : array<index_int, 3>{};
    const index_int n_idx  = tile_idx[0];
    const index_int th_idx = tile_idx[1];
    const index_int tw_idx = tile_idx[2];
    const int h0           = static_cast<int>(2 * th_idx) - 1;
    const int w0           = static_cast<int>(2 * tw_idx) - 1;
    const int32_t n_off    = static_cast<int32_t>(n_idx * x_sh[0]) * sizeof(half);
    const int32_t sh_b     = static_cast<int32_t>(x_sh[2] * sizeof(half));
    const int32_t sw_b     = static_cast<int32_t>(x_sh[3] * sizeof(half));
    // Per-lane H/W bounds for the V load. The X buffer descriptor only
    // checks [0, byte_count), so without these the boundary tiles silently
    // wrap into adjacent rows/channels. Precomputed once per lane (h0/w0 are
    // tile-fixed); also pre-evaluate the 4 per-i and 4 per-j masks so the cb
    // loop's per-(i, j) check is a single AND instead of two compares.
    const int v_i0       = -h0;
    const int v_i1       = static_cast<int>(in_h) - h0;
    const int v_j0       = -w0;
    const int v_j1       = static_cast<int>(in_w) - w0;
    const int32_t hw_off = h0 * sh_b + w0 * sw_b;
    const bool v_hok0    = (0 >= v_i0 and 0 < v_i1);
    const bool v_hok1    = (1 >= v_i0 and 1 < v_i1);
    const bool v_hok2    = (2 >= v_i0 and 2 < v_i1);
    const bool v_hok3    = (3 >= v_i0 and 3 < v_i1);
    const bool v_wok0    = (0 >= v_j0 and 0 < v_j1);
    const bool v_wok1    = (1 >= v_j0 and 1 < v_j1);
    const bool v_wok2    = (2 >= v_j0 and 2 < v_j1);
    const bool v_wok3    = (3 >= v_j0 and 3 < v_j1);

    // c-block range for this wave (= [0, cblocks) when SK=1; partitioned
    // round-robin across the SK waves of an NT-group when SK>1).
    const index_int cb_per_part = (cblocks + SK - 1) / SK;
    const index_int cb_start    = (SK == 1) ? index_int{0} : wave_sk_part * cb_per_part;
    const index_int cb_end_raw  = (SK == 1) ? cblocks : (cb_start + cb_per_part);
    const index_int cb_end      = (cb_end_raw < cblocks) ? cb_end_raw : cblocks;

    // Loop-invariant across the cb loop below: the OOB sentinel offset, the
    // fp16 zero, and the per-lane H/W in-bounds masks (h0/w0 are tile-fixed).
    const int32_t oob_byte  = static_cast<int32_t>(x_byte_count);
    const half hzero        = half(0.0f);
    const array<bool, 4> hi = {v_hok0, v_hok1, v_hok2, v_hok3};
    const array<bool, 4> wj = {v_wok0, v_wok1, v_wok2, v_wok3};

    for(index_int cb = cb_start; cb < cb_end; ++cb)
    {
        const index_int c_base = cb * CB;

        // ---- Per-lane V compute into registers ----
        // Each lane processes its own 8 c values (per v_chunk). The natural
        // lane mapping (lane%16 -> nt, lane/16*8 -> c_chunk_start) places the
        // V values right where the WMMA B operand expects them — no LDS round
        // trip and no cross-lane permute.
        //
        // Two-phase to expose load/transform overlap: issue every (vc, ci)
        // tile's d-load first (all into d_buf, constexpr-indexed so it stays
        // register-resident), then run the input transform on each. This
        // lets the global d-loads stay in flight while the transforms of
        // earlier tiles compute, hiding the load latency. The X buffer
        // descriptor only enforces a single [0, byte_count) extent — h-OOB
        // silently wraps into the next channel, w-OOB into the next row. For
        // sw_b == 2 we keep the fast b64 load and post-mask the w-OOB
        // columns; the per-element fallback handles other strides. Inactive
        // lanes have `off == x_byte_count`, so every load returns 0.
        if constexpr(NHWC)
        {
            // NHWC: each lane loads its tile's 8 channels per spatial position
            // with one b128 and runs the input transform vectorized over those
            // 8 channels.
            repeat_c<v_chunks>([&](auto vc_val) {
                constexpr index_int vc  = vc_val;
                const index_int c_start = c_base + vc * 16 + c_off;
                const bool active       = nt_active and (c_start < in_c);
                const int32_t base_off =
                    n_off + static_cast<int32_t>(c_start * x_sh[1]) * sizeof(half) + hw_off;
                const bool c_partial = (c_start + 8 > in_c);
                array<vec<half, 8>, 16> d_vec;
                repeat_c<4>([&](auto i_val) {
                    constexpr int i = i_val;
                    repeat_c<4>([&](auto j_val) {
                        constexpr int j   = j_val;
                        const bool ok     = active and hi[i] and wj[j];
                        const int32_t off = ok ? base_off + i * sh_b + j * sw_b : oob_byte;
                        auto v8           = buffer_load_half8(x_rsrc, off);
                        // Last channel block may be partial: zero c >= C, which
                        // would otherwise read the next pixel's channels.
                        if(c_partial)
                        {
                            repeat_c<8>([&](auto ci) {
                                if(c_start + index_int{ci} >= in_c)
                                    v8[index_int{ci}] = hzero;
                            });
                        }
                        d_vec[i * 4 + j] = v8;
                    });
                });
                array<vec<half, 8>, 16> v_vec;
                winograd_input_transform_f23_vec(d_vec, v_vec);
                repeat_c<8>([&](auto ci_val) {
                    constexpr index_int ci = ci_val;
                    repeat_c<16>([&](auto wp_val) {
                        constexpr index_int wp = wp_val;
                        v_lane[vc][ci][wp]     = v_vec[wp][index_int{ci}];
                    });
                });
            });
        }
        else
        {
            auto load_d = [&](index_int vc, index_int ci) {
                const index_int c_in_block = vc * 16 + c_off + ci;
                const index_int c          = c_base + c_in_block;
                const bool active          = nt_active and (c < in_c);
                const int32_t base_off = n_off + static_cast<int32_t>(c * x_sh[1]) * sizeof(half);
                const int32_t off      = active ? (base_off + hw_off) : oob_byte;
                array<half, 16> d;
                if(sw_b == 2)
                {
                    repeat_c<4>([&](auto i) {
                        const int32_t row_off = hi[i] ? off + static_cast<int>(i) * sh_b : oob_byte;
                        auto row              = buffer_load_half4(x_rsrc, row_off);
                        d[i * 4 + 0]          = wj[0] ? row.x : hzero;
                        d[i * 4 + 1]          = wj[1] ? row.y : hzero;
                        d[i * 4 + 2]          = wj[2] ? row.z : hzero;
                        d[i * 4 + 3]          = wj[3] ? row.w : hzero;
                    });
                }
                else
                {
                    repeat_c<4>([&](auto i) {
                        repeat_c<4>([&](auto j) {
                            const int32_t e_off =
                                (hi[i] and wj[j])
                                    ? off + static_cast<int>(i) * sh_b + static_cast<int>(j) * sw_b
                                    : oob_byte;
                            d[i * 4 + j] = buffer_load_half(x_rsrc, e_off);
                        });
                    });
                }
                return d;
            };
            array<array<array<half, 16>, 8>, v_chunks> d_buf;
            repeat_c<v_chunks>([&](auto vc_val) {
                constexpr index_int vc = vc_val;
                repeat_c<8>([&](auto ci_val) { d_buf[vc][ci_val] = load_d(vc, ci_val); });
            });
            repeat_c<v_chunks>([&](auto vc_val) {
                constexpr index_int vc = vc_val;
                repeat_c<8>([&](auto ci_val) {
                    constexpr index_int ci = ci_val;
                    auto v                 = winograd_input_transform_f23(d_buf[vc][ci]);
                    repeat_c<16>([&](auto wp) { v_lane[vc][ci][wp] = v[wp]; });
                });
            });
        }

        // ---- weight loader (dual mode) ----
        // FT (full transform): the weight is the raw filter g [3, 3, K, C] (9
        // halves per (k, c)); load_trows loads it and applies the row transform
        // T = G*g on the fly. !FT: the weight is the half-transformed T
        // [4, 3, K, C] (12 halves) and load_trows just loads the 4 T rows. In
        // both cases apply_gt then finishes the column transform U = T*G^T at
        // the WMMA dispatch. Buffer caching handles reuse across waves of a WG.
        //   T[0,j]=g0; T[1,j]=.5(g0+g1+g2); T[2,j]=.5(g0-g1+g2); T[3,j]=g2  (FT)
        //   U[i,0]=T0; U[i,1]=.5((T0+T2)+T1); U[i,2]=.5((T0+T2)-T1); U[i,3]=T2
        // u_sh strides are [3*K*C, K*C, C, 1] for either first-dim size (3 or
        // 4). t_off computes a byte offset, load_t does one b128 (8 fp16) load.
        static_assert(CB % 8 == 0, "CB must be a multiple of 8 for b128 U loads");
        auto t_off = [&](index_int i_g, index_int j_g, index_int k, index_int c_abs) {
            return static_cast<int32_t>(
                (i_g * u_sh[0] + j_g * u_sh[1] + k * u_sh[2] + c_abs * u_sh[3]) * sizeof(half));
        };
        auto load_t = [&](int32_t off) {
            auto raw = __builtin_amdgcn_raw_buffer_load_b128(u_rsrc, off, 0, 0);
            return bit_cast<vec<half, 8>>(raw);
        };
        struct u_row
        {
            vec<half, 8> u0, u1, u2, u3;
        };
        // One row of the row-transformed filter T = G*g: 3 columns.
        struct t_triple
        {
            vec<half, 8> t0, t1, t2;
        };
        auto apply_gt = [&](vec<half, 8> t0, vec<half, 8> t1, vec<half, 8> t2) -> u_row {
            const half half_c = 0.5f;
            // u0 = t0; u3 = t2; u1 = 0.5*((t0+t2) + t1); u2 = 0.5*((t0+t2) - t1)
            vec<half, 8> s = t0 + t2;
            return u_row{t0, (s + t1) * half_c, (s - t1) * half_c, t2};
        };
        // Return the 4 rows of T (each = 3 columns of 8 c values) for (k,
        // c_abs). FT loads the raw 3x3 g (9 b128 loads) and applies the row
        // transform G*; !FT loads the 4 stored T rows directly (12 loads). An
        // OOB k (>= K) reads zeros via the buffer descriptor.
        auto load_trows = [&](index_int k, index_int c_abs) {
            auto ld = [&](index_int i, index_int j) {
                const int32_t off =
                    (k < out_c) ? t_off(i, j, k, c_abs) : static_cast<int32_t>(u_byte_count);
                return load_t(off);
            };
            array<t_triple, 4> r;
            if constexpr(FT)
            {
                const vec<half, 8> g00 = ld(0, 0);
                const vec<half, 8> g01 = ld(0, 1);
                const vec<half, 8> g02 = ld(0, 2);
                const vec<half, 8> g10 = ld(1, 0);
                const vec<half, 8> g11 = ld(1, 1);
                const vec<half, 8> g12 = ld(1, 2);
                const vec<half, 8> g20 = ld(2, 0);
                const vec<half, 8> g21 = ld(2, 1);
                const vec<half, 8> g22 = ld(2, 2);
                const half hc          = 0.5f;
                r[0]                   = t_triple{g00, g01, g02};
                r[1]                   = t_triple{
                    (g00 + g10 + g20) * hc, (g01 + g11 + g21) * hc, (g02 + g12 + g22) * hc};
                r[2] = t_triple{
                    (g00 - g10 + g20) * hc, (g01 - g11 + g21) * hc, (g02 - g12 + g22) * hc};
                r[3] = t_triple{g20, g21, g22};
            }
            else
            {
                r[0] = t_triple{ld(0, 0), ld(0, 1), ld(0, 2)};
                r[1] = t_triple{ld(1, 0), ld(1, 1), ld(1, 2)};
                r[2] = t_triple{ld(2, 0), ld(2, 1), ld(2, 2)};
                r[3] = t_triple{ld(3, 0), ld(3, 1), ld(3, 2)};
            }
            return r;
        };
        // Cooperative U → LDS load for SK==1 (every wave reads the same U).
        // For SK>1 there is no cross-wave U sharing, so we skip this and
        // stream U directly from global into the WMMA A operand below.
        if constexpr(u_via_lds)
        {
            if constexpr(FT)
            {
                // g storage: one task per (k_idx, k_in_block, c_chunk) -- load
                // the raw 3x3 g once and write the full 4x4 U into LDS.
                constexpr index_int u_tasks = KW * bk * (CB / 8);
                idx.local_stride(_c<u_tasks>, [&](auto task) {
                    // task decomposes over {KW, bk, CB/8} -> (k_idx, k_in_block, c_half).
                    const auto t3              = array<index_int, 3>{KW, bk, CB / 8}.multi(task);
                    const index_int k_idx      = t3[0];
                    const index_int k_in_block = t3[1];
                    const index_int c_in_block = t3[2] * 8;
                    const index_int k          = k_base + k_idx * bk + k_in_block;
                    auto trows                 = load_trows(k, c_base + c_in_block);
                    repeat_c<4>([&](auto i_t_val) {
                        const index_int i_t = i_t_val;
                        auto ur             = apply_gt(trows[i_t].t0, trows[i_t].t1, trows[i_t].t2);
                        *as_vec<8>(
                            &u_smem[u_cache_idx(k_idx, i_t * 4 + 0, k_in_block, c_in_block)]) =
                            ur.u0;
                        *as_vec<8>(
                            &u_smem[u_cache_idx(k_idx, i_t * 4 + 1, k_in_block, c_in_block)]) =
                            ur.u1;
                        *as_vec<8>(
                            &u_smem[u_cache_idx(k_idx, i_t * 4 + 2, k_in_block, c_in_block)]) =
                            ur.u2;
                        *as_vec<8>(
                            &u_smem[u_cache_idx(k_idx, i_t * 4 + 3, k_in_block, c_in_block)]) =
                            ur.u3;
                    });
                });
            }
            else
            {
                // T storage: one task per (i_t, k_idx, k_in_block, c_chunk) --
                // load one T row's 3 columns and write its 4 U columns. The
                // extra i_t parallelism keeps more lanes issuing loads.
                constexpr index_int u_tasks = KW * 4 * bk * (CB / 8);
                idx.local_stride(_c<u_tasks>, [&](auto task) {
                    // task decomposes over {KW, 4, bk, CB/8} -> (k_idx, i_t, k_in_block, c_half).
                    const auto t4              = array<index_int, 4>{KW, 4, bk, CB / 8}.multi(task);
                    const index_int k_idx      = t4[0];
                    const index_int i_t        = t4[1];
                    const index_int k_in_block = t4[2];
                    const index_int c_in_block = t4[3] * 8;
                    const index_int k          = k_base + k_idx * bk + k_in_block;
                    vec<half, 8> t0;
                    vec<half, 8> t1;
                    vec<half, 8> t2;
                    if(k < out_c)
                    {
                        t0 = load_t(t_off(i_t, 0, k, c_base + c_in_block));
                        t1 = load_t(t_off(i_t, 1, k, c_base + c_in_block));
                        t2 = load_t(t_off(i_t, 2, k, c_base + c_in_block));
                    }
                    else
                    {
                        t0 = vec<half, 8>{0};
                        t1 = vec<half, 8>{0};
                        t2 = vec<half, 8>{0};
                    }
                    auto ur = apply_gt(t0, t1, t2);
                    *as_vec<8>(&u_smem[u_cache_idx(k_idx, i_t * 4 + 0, k_in_block, c_in_block)]) =
                        ur.u0;
                    *as_vec<8>(&u_smem[u_cache_idx(k_idx, i_t * 4 + 1, k_in_block, c_in_block)]) =
                        ur.u1;
                    *as_vec<8>(&u_smem[u_cache_idx(k_idx, i_t * 4 + 2, k_in_block, c_in_block)]) =
                        ur.u2;
                    *as_vec<8>(&u_smem[u_cache_idx(k_idx, i_t * 4 + 3, k_in_block, c_in_block)]) =
                        ur.u3;
                });
            }
            // Workgroup barrier so every wave sees the cooperative writes
            // before reading them for WMMA. NW==1 has only one wave so the
            // s_wait_dscnt the compiler inserts before the LDS read suffices.
            if constexpr(NW > 1)
                __syncthreads();
        }

        // ---- WMMA with fused incremental output transform ----
        constexpr index_int wmma_chunks = CB / 16;

        // Row-sum alpha fold: compute 4 wp (one full wp_i row of M) and apply
        // the A^T M A output transform via S0/S1 row sums into y[k_idx].
        auto fold_row = [&](auto k_idx_val,
                            auto wp_i_val,
                            const vec<float, 8>& m0,
                            const vec<float, 8>& m1,
                            const vec<float, 8>& m2,
                            const vec<float, 8>& m3) {
            // M*A column transform for this row i (= wp_i): produce the two
            // output-column partials as packed vec<float,8> over the 8 K rows.
            const vec<float, 8> s0 = m0 + m1 + m2;
            const vec<float, 8> s1 = m1 - m2 - m3;
            // A^T left multiply, accumulated incrementally into y. The
            // coefficients at[r][wp_i] are exactly 0 / +1 / -1, so the whole
            // 8-row update is one packed vec add/sub. With coef_r a constexpr +-1.0, the
            // `coef_r * s` multiply folds to an identity/negate at -O3, so
            // this is bit-identical to the scalar form. y[r*2+0] takes
            // output column c=0 (s0), y[r*2+1] takes column c=1 (s1).
            repeat_c<2>([&](auto r) {
                constexpr float coef_r = at[r][wp_i_val];
                if constexpr(coef_r != 0.f)
                {
                    y[k_idx_val][r * 2 + 0] = y[k_idx_val][r * 2 + 0] + coef_r * s0;
                    y[k_idx_val][r * 2 + 1] = y[k_idx_val][r * 2 + 1] + coef_r * s1;
                }
            });
        };
        // Construct V[wp] for this lane from register-resident v_lane.
        // c_offset selects which c-chunk (0 for CB=16; 0 or 16 for CB=32).
        // The 8 fp16 values come from v_lane[vc][0..7][wp] where vc = c_offset/16.
        auto load_v = [&](index_int wp, index_int c_offset) {
            const index_int vc = c_offset / 16;
            vec<half, 8> b;
            b.s0 = v_lane[vc][0][wp];
            b.s1 = v_lane[vc][1][wp];
            b.s2 = v_lane[vc][2][wp];
            b.s3 = v_lane[vc][3][wp];
            b.s4 = v_lane[vc][4][wp];
            b.s5 = v_lane[vc][5][wp];
            b.s6 = v_lane[vc][6][wp];
            b.s7 = v_lane[vc][7][wp];
            return b;
        };
        // A-operand source: either LDS (cooperative SK==1 path) or computed on
        // the fly from the weight (SK>1 path, via load_trows + apply_gt).
        auto load_u_row = [&](index_int k_idx, index_int wp_i, index_int c_offset) {
            if constexpr(u_via_lds)
            {
                u_row r;
                r.u0 = *as_vec<8>(
                    &u_smem[u_cache_idx(k_idx, wp_i * 4 + 0, m_in_wave, c_offset + c_off)]);
                r.u1 = *as_vec<8>(
                    &u_smem[u_cache_idx(k_idx, wp_i * 4 + 1, m_in_wave, c_offset + c_off)]);
                r.u2 = *as_vec<8>(
                    &u_smem[u_cache_idx(k_idx, wp_i * 4 + 2, m_in_wave, c_offset + c_off)]);
                r.u3 = *as_vec<8>(
                    &u_smem[u_cache_idx(k_idx, wp_i * 4 + 3, m_in_wave, c_offset + c_off)]);
                return r;
            }
            else
            {
                auto trows = load_trows(k_base + k_idx * bk + m_in_wave, c_base + c_offset + c_off);
                return apply_gt(trows[wp_i].t0, trows[wp_i].t1, trows[wp_i].t2);
            }
        };

        repeat_c<KW>([&](auto k_idx_val) {
            constexpr int k_idx = k_idx_val;
            // For the direct (SK>1) path: load the raw 3x3 g once per c-chunk
            // and apply the row transform up front (load_trows), staging the 4
            // T rows in t_buf. apply_gt (the column transform) then runs at the
            // WMMA dispatch, overlapping with the in-flight global loads.
            array<t_triple, wmma_chunks == 2 ? 8 : 4> t_buf;
            if constexpr(not u_via_lds)
            {
                const index_int kk = k_base + k_idx * bk + m_in_wave;
                auto tr0           = load_trows(kk, c_base + c_off);
                if constexpr(wmma_chunks == 2)
                {
                    auto tr1 = load_trows(kk, c_base + 16 + c_off);
                    repeat_c<4>([&](auto wp_i_val) {
                        const index_int wp_i = wp_i_val;
                        t_buf[wp_i * 2 + 0]  = tr0[wp_i];
                        t_buf[wp_i * 2 + 1]  = tr1[wp_i];
                    });
                }
                else
                {
                    repeat_c<4>([&](auto wp_i_val) {
                        const index_int wp_i = wp_i_val;
                        t_buf[wp_i]          = tr0[wp_i];
                    });
                }
            }
            auto get_ur = [&](auto wp_i_val, index_int c_offset) {
                constexpr int wp_i = wp_i_val;
                if constexpr(u_via_lds)
                {
                    return load_u_row(k_idx, wp_i, c_offset);
                }
                else
                {
                    const auto& tt = (wmma_chunks == 2) ? t_buf[wp_i * 2 + (c_offset == 16 ? 1 : 0)]
                                                        : t_buf[wp_i];
                    return apply_gt(tt.t0, tt.t1, tt.t2);
                }
            };
            repeat_c<4>([&](auto wp_i_val) {
                constexpr int wp_i = wp_i_val;
                vec<float, 8> m0{};
                vec<float, 8> m1{};
                vec<float, 8> m2{};
                vec<float, 8> m3{};
                if constexpr(wmma_chunks == 2)
                {
                    auto u_lo = get_ur(wp_i_val, 0);
                    auto b0   = load_v(wp_i * 4 + 0, 0);
                    auto b1   = load_v(wp_i * 4 + 1, 0);
                    auto b2   = load_v(wp_i * 4 + 2, 0);
                    auto b3   = load_v(wp_i * 4 + 3, 0);
                    auto u_hi = get_ur(wp_i_val, 16);
                    auto b4   = load_v(wp_i * 4 + 0, 16);
                    auto b5   = load_v(wp_i * 4 + 1, 16);
                    auto b6   = load_v(wp_i * 4 + 2, 16);
                    auto b7   = load_v(wp_i * 4 + 3, 16);
                    vec<float, 8> m4{};
                    vec<float, 8> m5{};
                    vec<float, 8> m6{};
                    vec<float, 8> m7{};
                    wmma_octet_asm(u_lo.u0,
                                   b0,
                                   u_lo.u1,
                                   b1,
                                   u_lo.u2,
                                   b2,
                                   u_lo.u3,
                                   b3,
                                   u_hi.u0,
                                   b4,
                                   u_hi.u1,
                                   b5,
                                   u_hi.u2,
                                   b6,
                                   u_hi.u3,
                                   b7,
                                   m0,
                                   m1,
                                   m2,
                                   m3,
                                   m4,
                                   m5,
                                   m6,
                                   m7);
                    m0 = m0 + m4;
                    m1 = m1 + m5;
                    m2 = m2 + m6;
                    m3 = m3 + m7;
                }
                else
                {
                    for(index_int ck = 0; ck < wmma_chunks; ++ck)
                    {
                        const index_int c_offset = ck * 16;
                        auto ur                  = get_ur(wp_i_val, c_offset);
                        auto b0                  = load_v(wp_i * 4 + 0, c_offset);
                        auto b1                  = load_v(wp_i * 4 + 1, c_offset);
                        auto b2                  = load_v(wp_i * 4 + 2, c_offset);
                        auto b3                  = load_v(wp_i * 4 + 3, c_offset);
                        wmma_quad_asm(ur.u0, b0, ur.u1, b1, ur.u2, b2, ur.u3, b3, m0, m1, m2, m3);
                    }
                }
                fold_row(_c<k_idx>, _c<wp_i>, m0, m1, m2, m3);
            });
        });

        // End-of-cb barrier for the SK==1 cooperative path: the next iter
        // overwrites the WG-shared U slots, so all waves must finish their
        // WMMA reads first. NW==1 has only one wave; SK>1 streams U direct
        // and doesn't touch the LDS slots, so neither needs the sync.
        if constexpr(u_via_lds and NW > 1)
            __syncthreads();
    }

    // ---- Split-c cross-wave reduce (SK>1 only) ----
    // After the per-wave c-block loop each wave holds a y_partial that covers
    // only its assigned 1/SK of the c contraction. Sum across the SK waves of
    // an NT-group via LDS so the wave_sk_part==0 wave ends up with the full
    // y, then have that wave do the writeback. SK=1 skips this entirely.
    if constexpr(SK > 1)
    {
        // Layout: y_reduce_lds[wave_nt_idx][wave_sk_part][lane][output(0..3)][ki(0..7)]
        // KW=1 enforced by static_assert when SK>1.
        constexpr index_int per_lane_floats = 32; // 4 outputs * 8 ki
        const index_int wave_red_off = (wave_nt_idx * SK + wave_sk_part) * 32 * per_lane_floats;
        const index_int lane_off     = wave_red_off + lane * per_lane_floats;
        // y[0][out_i] is one vec<float,8>; the 8 ki land contiguously in LDS
        // (lane_off and out_i*8 are both multiples of 8 floats -> 32-byte
        // aligned), so move each output as one packed vec store/load.
        repeat_c<4>(
            [&](auto out_i) { *as_vec<8>(&y_reduce_lds[lane_off + out_i * 8]) = y[0][out_i]; });
        __syncthreads();
        // wave_sk_part == 0 waves sum partials from waves 1..SK-1 of their
        // NT-group into their own y; other waves are now idle for writeback.
        if(wave_sk_part == 0)
        {
            const index_int group_base = wave_nt_idx * SK * 32 * per_lane_floats;
            for(index_int s = 1; s < SK; ++s)
            {
                const index_int s_off =
                    group_base + s * 32 * per_lane_floats + lane * per_lane_floats;
                repeat_c<4>([&](auto out_i) {
                    y[0][out_i] = y[0][out_i] + *as_vec<8>(&y_reduce_lds[s_off + out_i * 8]);
                });
            }
        }
    }

    // ---- Output writeback for each k_block this workgroup covered ----
    // Reuse the per-lane (n_idx, th_idx, tw_idx) computed up at V-load setup.
    // For SK>1 only the wave_sk_part==0 wave of each NT-group has the summed y.
    using out_type               = typename Output::type;
    const index_int k_row_offset = c_off; // (lane / 16) * 8, same lane mapping
    if(not nt_active)
        return;
    if constexpr(SK > 1)
    {
        if(wave_sk_part != 0)
            return;
    }

    const auto sn  = out_shape.strides[0];
    const auto sk  = out_shape.strides[1];
    const auto sh  = out_shape.strides[2];
    const auto sw  = out_shape.strides[3];
    auto* out_data = output.data();

    // Fused post-op: apply f(y_val, inputs[multi_idx]...) at each output
    // position. For the non-fused case, F = op::id{} and Inputs... is empty
    // so the call collapses to `static_cast<out_type>(y_val)`. For fused
    // pointwise, F is the generated post_winograd_conv function and the
    // extra inputs are indexed at the same (n, k, h_out, w_out) position as
    // the output.
    //
    // For NCHW + stride_w=1, pack the (j=0, j=1) outputs of each 2x2 winograd
    // tile into a vec<out_type, 2> and write via one store. Without this the
    // compiler emits two narrow stores per pair when there's a non-trivial
    // post-op `f` (it stops packing the converted j=0/j=1 store pair). That
    // doubled the global_store count in fused kernels (e.g., a 192x192 conv
    // with bias+leaky_relu went 86us unfused → 122us fused).
    //
    // Enabled for any 2-byte output type (fp16, bf16): two of them pack into
    // a single b32 store. The output type is whatever the fused pointwise
    // converts to — NOT assumed to be fp16 — and each extra input is packed
    // at its own element type, so the result matches the slow path exactly.
    // Wider output types (fp32, etc.) fall through to the slow path.
    if constexpr(NHWC)
    {
        // NHWC output (sk == 1): the 8 K values of a lane are contiguous, so
        // pack them into one b128 store per output spatial position. K not a
        // multiple of 8 (rare) falls back to scalar stores; K-OOB is masked.
        // When the output is still NCHW (e.g. the standardized program output),
        // fall through to the packed NCHW store below.
        if(sk == 1)
        {
            const bool k_pack = (out_c % 8 == 0);
            repeat_c<KW>([&](auto k_idx_val) {
                constexpr int k_idx     = k_idx_val;
                const index_int k_first = k_base + k_idx * bk + k_row_offset;
                if(k_first >= out_c)
                    return;
                repeat_c<2>([&](auto i_val) {
                    constexpr int i       = i_val;
                    const index_int h_out = 2 * th_idx + i;
                    if(h_out >= out_h)
                        return;
                    repeat_c<2>([&](auto j_val) {
                        constexpr int j       = j_val;
                        const index_int w_out = 2 * tw_idx + j;
                        if(w_out >= out_w)
                            return;
                        const index_int oidx = i * 2 + j;
                        const index_int off  = n_idx * sn + k_first * sk + h_out * sh + w_out * sw;
                        if(k_pack)
                        {
                            vec<PostInput, 8> yv;
                            repeat_c<8>([&](auto ki) {
                                yv[index_int{ki}] =
                                    static_cast<PostInput>(y[k_idx][oidx][index_int{ki}]);
                            });
                            // Build a vec<8> of each fused input over the 8 K (which
                            // are contiguous in K), so f runs packed and returns one
                            // vec<out_type, 8> for a single b128 store.
                            auto in_vec8 = [&](const auto& inp) {
                                vec<remove_reference_t<decltype(inp[array<index_int, 4>{}])>, 8> iv;
                                repeat_c<8>([&](auto ki) {
                                    iv[index_int{ki}] = inp[array<index_int, 4>{
                                        n_idx, k_first + index_int{ki}, h_out, w_out}];
                                });
                                return iv;
                            };
                            *as_vec<8>(&out_data[off]) = f(yv, in_vec8(inputs)...);
                        }
                        else
                        {
                            repeat_c<8>([&](auto ki) {
                                const index_int k = k_first + index_int{ki};
                                if(k < out_c)
                                {
                                    const array<index_int, 4> oid{n_idx, k, h_out, w_out};
                                    output[oid] = static_cast<out_type>(
                                        f(static_cast<PostInput>(y[k_idx][oidx][index_int{ki}]),
                                          inputs[oid]...));
                                }
                            });
                        }
                    });
                });
            });
            return;
        }
    }
    constexpr bool packed_store_ok = sizeof(out_type) == 2;
    repeat_c<KW>([&](auto k_idx_val) {
        constexpr int k_idx         = k_idx_val;
        const index_int base_offset = n_idx * sn + (k_base + k_idx * bk + k_row_offset) * sk +
                                      (2 * th_idx) * sh + (2 * tw_idx) * sw;
        const bool w_pair_in = (2 * tw_idx + 1 < out_w) and (sw == 1);
        repeat_c<8>([&](auto ki) {
            const index_int k = k_base + k_idx * bk + k_row_offset + ki;
            if(k < out_c)
            {
                const index_int kbase = base_offset + ki * sk;
                repeat_c<2>([&](auto i) {
                    const index_int h_out = 2 * th_idx + i;
                    if(h_out < out_h)
                    {
                        const index_int hbase = kbase + i * sh;
                        // Fast path: both W in-bounds and stride_w=1 — pack
                        // the two j outputs into one packed store.
                        if constexpr(packed_store_ok)
                        {
                            if(w_pair_in)
                            {
                                const index_int w_out0 = 2 * tw_idx;
                                const index_int w_out1 = w_out0 + 1;
                                const array<index_int, 4> idx0{n_idx, k, h_out, w_out0};
                                const array<index_int, 4> idx1{n_idx, k, h_out, w_out1};
                                // Pack the j=0 / j=1 pair into vec<out_type, 2>
                                // and call f once on the packed value. The
                                // generated post_winograd_conv function is
                                // templated on input types; invoking it with
                                // packed vecs lets the fused ops (add, mul,
                                // max) emit v_pk_* packed ops instead of two
                                // scalar ones. Each extra input is packed at
                                // its OWN element type — not forced to the
                                // output type — so f sees exactly what it
                                // sees in the slow path. The conv result is
                                // cast to PostInput (conv precision, or the
                                // pointwise's leading-convert target); f
                                // converts the result to out_type, yielding
                                // vec<out_type, 2> for one packed store.
                                vec<PostInput, 2> y_pair{
                                    static_cast<PostInput>(y[k_idx][i * 2 + 0][index_int{ki}]),
                                    static_cast<PostInput>(y[k_idx][i * 2 + 1][index_int{ki}])};
                                *as_vec<2>(&out_data[hbase]) =
                                    f(y_pair,
                                      vec<remove_reference_t<decltype(inputs[idx0])>, 2>{
                                          inputs[idx0], inputs[idx1]}...);
                                return;
                            }
                        }
                        // Slow path: scalar per-j store with bounds checks.
                        repeat_c<2>([&](auto j) {
                            const index_int w_out = 2 * tw_idx + j;
                            if(w_out < out_w)
                            {
                                const array<index_int, 4> out_idx{n_idx, k, h_out, w_out};
                                output[out_idx] = static_cast<out_type>(
                                    f(static_cast<PostInput>(y[k_idx][i * 2 + j][index_int{ki}]),
                                      inputs[out_idx]...));
                            }
                        });
                    }
                });
            }
        });
    });
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP
