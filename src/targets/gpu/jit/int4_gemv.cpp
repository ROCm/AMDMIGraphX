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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/env.hpp>
#include <string>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT4_GEMV_CONFIG);

// Config table ported from HIP EP's offline autotuned LUT (gfx1151, commit
// 13f44fb8).  TILE_N capped at 16 -- HIP EP proved 32/64 always spill and
// lose (732-1436 B scratch, 1.13x-9.3x slower across 50 sweeps).
//
// The function signature adds K so the table can distinguish shapes that share
// N but differ in reduction length (e.g. gate_up vs down_proj).
static std::pair<int, int> get_gemv_config(std::size_t n, std::size_t k,
                                           const std::string& arch)
{
    // Env override.  Two accepted forms:
    //
    //   "256,4"                      -- global: this config for every shape
    //   "3072:8192=256,16;9216:3072=256,4[;*=128,8]"
    //                                -- per-shape: N:K=BS,TN, semicolon separated,
    //                                   with an optional "*=BS,TN" default for
    //                                   shapes not named.
    //
    // The per-shape form exists because the global form makes per-shape optima
    // unmeasurable: one whole-model run reports four shapes, but they all ran
    // the same config, so finding the best config PER shape needed a rebuild per
    // candidate.  With this, one run can carry a different config per shape and
    // the per-op report separates them.
    //
    // Unparseable entries are skipped rather than throwing -- a typo in a sweep
    // script should fall through to the shipping table, not abort compilation.
    auto config_str = string_value_of(MIGRAPHX_INT4_GEMV_CONFIG{});
    if(not config_str.empty())
    {
        auto parse_pair = [](const std::string& s) -> std::pair<int, int> {
            auto comma = s.find(',');
            if(comma == std::string::npos)
                return {0, 0};
            try
            {
                return {std::stoi(s.substr(0, comma)), std::stoi(s.substr(comma + 1))};
            }
            catch(...)
            {
                return {0, 0};
            }
        };

        if(config_str.find('=') == std::string::npos)
        {
            auto cfg = parse_pair(config_str);
            if(cfg.first > 0 and cfg.second > 0)
                return cfg;
        }
        else
        {
            std::pair<int, int> fallback{0, 0};
            std::size_t pos = 0;
            while(pos <= config_str.size())
            {
                auto semi  = config_str.find(';', pos);
                auto entry = config_str.substr(pos, semi - pos);
                pos        = (semi == std::string::npos) ? config_str.size() + 1 : semi + 1;
                if(entry.empty())
                    continue;

                auto eq = entry.find('=');
                if(eq == std::string::npos)
                    continue;
                auto key = entry.substr(0, eq);
                auto cfg = parse_pair(entry.substr(eq + 1));
                if(cfg.first <= 0 or cfg.second <= 0)
                    continue;

                if(key == "*")
                {
                    fallback = cfg;
                    continue;
                }
                auto colon = key.find(':');
                if(colon == std::string::npos)
                    continue;
                try
                {
                    if(std::stoull(key.substr(0, colon)) == n and
                       std::stoull(key.substr(colon + 1)) == k)
                        return cfg;
                }
                catch(...)
                {
                    continue;
                }
            }
            if(fallback.first > 0)
                return fallback;
        }
    }

    // --------------------------------------------------------------------
    // v6: split on K at 2048 FIRST.
    //
    // v4's table had no K threshold below 4096 in its large-N branches, so
    // Qwen's gate/up (N=4864, K=896) and Phi's gate/up (N=8192, K=3072)
    // both fell through to {32,4} -- the same config for a 3.4x difference
    // in reduction length.  Measured on STX-Halo that single collision is
    // worth +7% on Qwen and -40% on Phi simultaneously, which is why no
    // single-table version beat baseline on both.  The lm_head bucket
    // (n>=16384) collided the same way: Qwen K=896 and Phi K=3072 both
    // landed on {128,2}.
    //
    // Short reductions cannot feed a wide block: at K=896 a BLOCK_SIZE of
    // 256 leaves most lanes with <4 elements each, so the small-BS entries
    // that HIP EP autotuned are right there.  At K>=2048 there is enough
    // reduction work to fill a wide block, and v3's larger BS/TN entries
    // measured +56% on Phi where v4's small blocks measured -40%.
    //
    // So: K < 2048 keeps v4's HIP-EP-autotuned entries, K >= 2048 uses v3's
    // N-keyed entries.  Both halves are the configuration that was actually
    // measured best for the shapes that reach it -- this is not a
    // compromise between the two tables, it is the union of their wins.
    // --------------------------------------------------------------------

    // v7 addendum: the vocabulary projection is its own regime, keyed on N alone.
    //
    // An lm_head is structurally unlike a transformer-block matmul: N is the
    // vocab size (100k+), K is modest, and it runs once per token with no
    // reuse across the N dimension.  TILE_N=32 there asks each thread to hold
    // 32 accumulators across a 128k-wide output -- the case the HIP EP sweep
    // referenced above found always spills.
    //
    // v6 routed this correctly for Qwen only by accident: its lm_head is
    // K=896, so it fell into the short-reduction branch and got {128,2}.
    // Llama's lm_head is K=2048, so it fell out the other side onto {256,32}.
    // Both models that own a huge-N lm_head (Qwen 151936, Llama 128256)
    // regress under a table that gives it TN=32; Phi, whose graph has NO
    // quantized lm_head at all, is the one model that gains under v3.
    // Make the routing explicit instead of K-dependent.
    //
    // NOTE (2026-09-14): shape-level reasoning, not yet a per-op measurement.
    // If the profile shows lm_head is not a material share of Llama decode,
    // this branch is wrong and should be reverted, not tuned.
    //
    // v8 CORRECTION (2026-09-14, from the STX-Halo driver profile + kernel analysis).
    // Two things above are wrong and both were load-bearing:
    //
    // (1) This is NOT an "lm_head branch".  MIGraphX concat-fuses gate+up before
    //     the GEMV, so Llama's and Phi's fused gate_up are BOTH N=16384 and land
    //     here too.  On Llama that is ~43% of weight traffic on top of the
    //     lm_head's ~21% -- which is what actually explains v7's size, not the
    //     lm_head alone.  Qwen's gate_up fuses to 9728 and does NOT reach here.
    // (2) The lm_head is not slow at TILE_N=32.  Measured in the driver, Llama's
    //     128256x2048 runs 19.8% FASTER under v3's {256,32} than under baseline.
    //     The huge-N/TN=32 story is falsified; what v7 changed that mattered was
    //     BLOCK_SIZE, not TILE_N.
    //
    // (3) FALSIFIED BY MEASUREMENT -- kept because it was the reasoning that
    //     produced the experiment below, and a dead mechanism is only useful
    //     with its falsifier attached.  The argument was: the kernel's shuffle
    //     reduction runs unconditionally on every thread, while a thread only
    //     enters the K loop if tid < K/32, so reduction work per useful FMA is
    //     ~5*BLOCK_SIZE/K, TILE_N cancels, and BLOCK_SIZE above K/32 buys
    //     nothing but idle waves.  That predicts {64,8} beats {128,2} at
    //     K=2048.  Measured: it loses by ~6%.  See the v8 block below.
    //
    // ------------------------------------------------------------------
    // v8 MEASURED AND REVERTED (2026-09-14).  The prediction above was
    // signed before the run: "if v8 does not beat v7 on Llama, the 5*BS/K
    // model is wrong and this should revert to {128,2}."  It did not.
    //
    //   Llama, STX-Halo, 128 tok, tp_steady P50, cold cache, md5-verified:
    //     baseline          78.2 / 69.9
    //     v7 {128,2}       165.8 / 164.4 / 165.7   <- tight, both metrics agree
    //     v8 {64,8}        155.6  (5 iters, 152.8-156.5)
    //
    // v8 is ~6% below v7 and does not overlap its band.  So BLOCK_SIZE is
    // NOT simply capped by K/32: {64,8} cuts the modelled reduction ratio 4x
    // versus v7 and measures SLOWER, which the 5*BS/K model cannot produce.
    //
    // Reverting to {128,2} as pre-registered rather than tuning around the
    // result -- the discriminating experiment was chosen so that a loss kills
    // the mechanism, and a mechanism that is dead should not be re-fitted.
    // NOTE this leaves v7's own explanation UNSETTLED: {128,2} is the config
    // that measured best, not a config we can currently explain.  What is
    // established is that the branch catches fused gate_up (N=16384) as well
    // as the lm_head, and that TILE_N=32 is not the problem (the lm_head is
    // 19.8% faster with it).
    // ------------------------------------------------------------------
    // --------------------------------------------------------------------
    // v12: measured-winner allowlist, keyed on (arch, N, K) EXACTLY.
    //
    // Everything below this block is a heuristic -- a set of ranges fitted to
    // a handful of shapes and then applied to every shape that happens to fall
    // in the range.  That is how the v6 collision happened (Qwen K=896 and Phi
    // K=3072 landing on one entry) and how the n>=16384 branch came to cost
    // Navi48 7.3% while helping STX-Halo.
    //
    // This table is the opposite construction, and it has three properties
    // that the ranges cannot have:
    //
    //   * No regression by construction.  An unmeasured (arch,n,k) cannot
    //     reach a new value -- it falls through byte-for-byte to the heuristic
    //     that shipped before this block existed.
    //   * `arch` is in the key, so a winner measured on gfx1151 cannot be
    //     imposed on gfx1201.  The parameter was already accepted and ignored;
    //     this is what it was there for.
    //   * Exact match, not ranges, so adding a row affects exactly the one
    //     shape it was measured on and nothing else.
    //
    // Every row must cite the measurement that produced it.  A row without a
    // measurement is a heuristic wearing a table's clothing, which is the
    // thing this block exists to stop.
    // --------------------------------------------------------------------
    struct gemv_entry
    {
        const char* arch;
        std::size_t n;
        std::size_t k;
        int block_size;
        int tile_n;
    };
    // Phi-3.5-mini INT4 AWQ block128 decode shapes.  Whole-model wall median on
    // STX-Halo (gfx1151), migraphx-driver perf -n 50: {256,4} = 12.2554 ms vs
    // 16.7416 ms for the heuristic below (-26.8%).  Gate: token ids identical.
    //
    // Llama-3.2-1B-Instruct INT4 decode shapes (K=2048 family).  End-to-end
    // Throughput Steady P50 on STX-Halo (gfx1151), ep.v13-dot2, 128 new / 256
    // max_length, 5 iters, cold .mxr per arm, TIMING VALIDITY: valid.  Gate:
    // token ids identical to baseline on every arm quoted.
    //
    //   heuristic below (falls to {128,2} via n>=16384, {128,16} otherwise)
    //                                    167.77  <- what shipped before this
    //   all five at {256,4}              219.85  (+31.0%)
    //   ... + gate_up at {64,4}          229.29  (+36.7%)   <- this table
    //
    // gate_up (16384x2048) DOMINATES: moving only it to {256,2} costs 32.9%
    // end-to-end, while every other shape costs 3-8%.  It is the only shape
    // here that was bracketed on both axes -- BLOCK_SIZE 32/64/128/256 =
    // 164.32 / 229.29 / 226.50 / 219.85, and TILE_N 2/4/8 at BS=64 =
    // 225.98 / 229.29 / 211.79.  Two cliffs (BS=32, and TN=2 at BS=256);
    // {64,4} sits between them.
    //
    // The other four rows are best-of-measured, NOT bracketed: each was moved
    // to {256,16} (and lm_head also to {128,2}) and lost, so {256,4} is the
    // best config tried, not a proven local optimum.  Kept as a row anyway --
    // it beats the heuristic by a measured margin -- but do not read them as
    // settled.  See NEXT-CONFIG-AUTOTUNE.md (T5) for the cheap way to finish
    // bracketing them.
    //
    // NOTE the sign conflict with phi3.5 above: phi's 16384x3072 gate_up wants
    // BLOCK_SIZE 256, Llama's 16384x2048 gate_up wants 64.  Same op, same arch,
    // adjacent K -- and opposite directions.  BLOCK_SIZE has no fixed sign; do
    // not "simplify" these two rows into a range.
    static const gemv_entry measured[] = {
        {"gfx1151", 16384, 3072, 256, 4}, // phi3.5  fused gate_up
        {"gfx1151", 9216, 3072, 256, 4},  // phi3.5  fused qkv
        {"gfx1151", 3072, 8192, 256, 4},  // phi3.5  down_proj
        {"gfx1151", 3072, 3072, 256, 4},  // phi3.5  o_proj
        {"gfx1151", 16384, 2048, 64, 4},  // llama1b fused gate_up (bracketed)
        {"gfx1151", 3072, 2048, 256, 4},  // llama1b fused qkv
        {"gfx1151", 2048, 2048, 256, 4},  // llama1b o_proj
        {"gfx1151", 2048, 8192, 256, 4},  // llama1b down_proj
        {"gfx1151", 128256, 2048, 256, 4} // llama1b lm_head (block_k=32, no zp)
    };
    for(const auto& e : measured)
    {
        if(n == e.n and k == e.k and arch == e.arch)
            return {e.block_size, e.tile_n};
    }

    if(n >= 16384)
        return {128, 2};

    // Short-reduction regime (Qwen-class: K < 2048).  v4 / HIP EP entries.
    if(k < 2048)
    {
        if(n >= 4096)
            return {32, 4};
        return {32, 2};
    }

    // Long-reduction regime (Phi/Llama-class: K >= 2048).  v3 entries.
    if(n > 4096)
        return {256, 32};
    if(n > 1024)
        return {128, 16};
    return {64, 8};
}

// The kernel source receives void* pointers (MIGraphX convention), casts to typed pointers,
// then calls the device-side int4_gemv_kernel with compile-time N and K baked in.
static const char* const int4_gemv_kernel_src = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/int4_gemv.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void int4_gemv_main(${params})
{
    // arg0: A (fp16), arg1: B_packed (uint8), arg2: scales (fp16),
    // arg3: zero_points (uint8) [only if HAS_ZP],
    // next: bias (fp16) [only if HAS_BIAS], last arg: output (fp16)
    auto* a_ptr      = reinterpret_cast<const _Float16*>(private_p0);
    auto* b_ptr      = reinterpret_cast<const migraphx::uint8_t*>(private_p1);
    auto* scales_ptr = reinterpret_cast<const _Float16*>(private_p2);
    ${zp_cast}
    ${bias_cast}
    auto* out_ptr    = reinterpret_cast<_Float16*>(${out_arg});

    int4_gemv_kernel<${block_size}, ${tile_n}, ${block_k}, ${has_zp}, ${n_blocks}, ${has_bias}>(
        a_ptr, b_ptr, scales_ptr, zp_ptr, bias_ptr, out_ptr, ${N}u, ${K}u);
}

}

} // namespace migraphx

)__migraphx__";

struct int4_gemv_compiler : compiler<int4_gemv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::int4_gemv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // inputs: [A(f16), B_packed(u8), scales(f16), [zp(u8),] [bias(f16),] output(f16)]
        auto N        = v.at("N").to<std::size_t>();
        auto K        = v.at("K").to<std::size_t>();
        auto block_k  = v.at("block_k").to<int>();
        auto has_zp   = v.at("has_zp").to<bool>();
        auto has_bias = v.at("has_bias").to<bool>();

        auto [gemv_block_size, gemv_tile_n] = get_gemv_config(N, K, ctx.get_current_device().get_gfx_name());

        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "int4_gemv_main";

        // Flat 1D grid: total_blocks = n_blocks * batch
        // Kernel recovers: n_block = blockIdx.x % n_blocks, batch_id = blockIdx.x / n_blocks
        auto batch    = inputs.front().lens().front();
        auto n_blocks = (N + gemv_tile_n - 1) / gemv_tile_n;
        options.global = n_blocks * batch * gemv_block_size;
        options.local  = gemv_block_size;

        auto n_params = inputs.size();
        // Argument layout: p0=A, p1=B, p2=scales, then optionally zp, then optionally bias, then output
        int next_arg = 3;

        std::string zp_cast;
        if(has_zp)
        {
            zp_cast = "auto* zp_ptr = reinterpret_cast<const migraphx::uint8_t*>(private_p" + std::to_string(next_arg) + ");";
            next_arg++;
        }
        else
        {
            zp_cast = "const migraphx::uint8_t* zp_ptr = nullptr;";
        }

        std::string bias_cast;
        if(has_bias)
        {
            bias_cast = "auto* bias_ptr = reinterpret_cast<const _Float16*>(private_p" + std::to_string(next_arg) + ");";
            next_arg++;
        }
        else
        {
            bias_cast = "const _Float16* bias_ptr = nullptr;";
        }

        std::string out_arg = "private_p" + std::to_string(next_arg);

        auto src = interpolate_string(
            int4_gemv_kernel_src,
            {{"params", enum_params(n_params, "void * private_p")},
             {"block_size", std::to_string(gemv_block_size)},
             {"tile_n", std::to_string(gemv_tile_n)},
             {"block_k", std::to_string(block_k)},
             {"has_zp", has_zp ? "true" : "false"},
             {"has_bias", has_bias ? "true" : "false"},
             {"zp_cast", zp_cast},
             {"bias_cast", bias_cast},
             {"out_arg", out_arg},
             {"N", std::to_string(N)},
             {"K", std::to_string(K)},
             {"n_blocks", std::to_string(n_blocks)}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
