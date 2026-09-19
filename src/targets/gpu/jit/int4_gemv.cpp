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
#include <migraphx/optional.hpp>
#include <migraphx/instruction.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT4_GEMV_CONFIG);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT4_GEMV_TRACE);

// Config selection runs in three tiers, in this order:
//
//   1. MIGRAPHX_INT4_GEMV_CONFIG env override  -- parse_env_config(), below
//   2. measured[] exact-match table            -- get_gemv_config()
//   3. heuristic ladder                        -- get_gemv_config()
//
// Tiers 2 and 3 both live inside get_gemv_config(); it reports which one
// answered through its `trusted` out-param.
//
// Tiers 1 and 2 are trusted answers: a shape that hits either needs no search,
// and get_tuning_config() returns a single solution for it so compile_ops
// skips benchmarking entirely.  Tier 3 is a GUESS, and it is measured to lose
// 13-27% on shapes nobody tuned -- so a tier-3 shape gets the candidate grid
// benchmarked instead, with the heuristic pick seeded first so that behaviour
// is unchanged when benchmarking is skipped.
static optional<std::pair<int, int>> parse_env_config(std::size_t n, std::size_t k)
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
    return nullopt;
}

// Config for (n,k,arch).  When `trusted` is non-null it is set to true if the
// answer came from the env override or the measured[] table, and false if it
// came from the heuristic ladder -- i.e. false means "this is a guess".
static std::pair<int, int>
get_gemv_config(std::size_t n, std::size_t k, const std::string& arch, bool* trusted = nullptr)
{
    auto set_trusted = [&](bool t) {
        if(trusted != nullptr)
            *trusted = t;
    };

    if(auto env = parse_env_config(n, k))
    {
        set_trusted(true);
        return *env;
    }

    // ------------------------------------------------------------------
    // Config selection, in three tiers: env override, a table of measured
    // winners keyed on (arch, N, K) exactly, then a heuristic ladder.
    //
    // Why the table is exact-match rather than ranges.  Range-fitted
    // heuristics collide: shapes with very different reduction lengths land
    // on one entry and the config that is right for one is badly wrong for
    // the other.  Measured example -- Qwen's gate_up (N=4864, K=896) and
    // Phi's (N=8192, K=3072) shared an entry, and a single config for both
    // was simultaneously +7% on one model and -40% on the other.  An exact
    // key cannot do that: an unmeasured shape falls through to the heuristic
    // unchanged, so adding a row can only affect the shape it was measured
    // on.
    //
    // Why `arch` is in the key.  Winners do not transfer between
    // architectures -- the n>=16384 heuristic branch helps gfx1151 and costs
    // gfx1201 7.3%.  Keying on arch keeps a measurement from being imposed
    // on hardware it was never taken on.
    //
    // Why the ladder splits on K at 2048.  Short reductions cannot fill a
    // wide block: at K=896 a BLOCK_SIZE of 256 leaves most lanes with fewer
    // than four elements, so small blocks win.  At K>=2048 there is enough
    // reduction work to fill a wide block and the larger BLOCK_SIZE/TILE_N
    // entries win instead.
    //
    // Note on the huge-N branch: it is not an "lm_head branch".  MIGraphX
    // concat-fuses gate+up before the GEMV, so a fused gate_up also lands at
    // N=16384 and is typically the larger share of weight traffic.  Both
    // shapes reach this branch and they do not want the same config.
    //
    // BLOCK_SIZE has no fixed sign.  Phi's 16384x3072 gate_up wants 256;
    // Llama's 16384x2048 wants 64 -- same op, same arch, adjacent K,
    // opposite directions.  Do not collapse such rows into a range.
    //
    // When the table misses and `get_tuning_config` is consulted, the
    // heuristic result becomes the first candidate and the remaining grid is
    // benchmarked against it, so a miss costs compile time rather than
    // throughput.
    // ------------------------------------------------------------------

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
    // Throughput Steady P50 on STX-Halo (gfx1151), 128 new / 256
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
        {
            set_trusted(true);
            return {e.block_size, e.tile_n};
        }
    }

    // Everything below is the guess.
    set_trusted(false);

    if(n >= 16384)
        return {128, 2};

    // Short-reduction regime (Qwen-class: K < 2048).
    if(k < 2048)
    {
        if(n >= 4096)
            return {32, 4};
        return {32, 2};
    }

    // Long-reduction regime (Phi/Llama-class: K >= 2048).
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

        // A solution supplied by the tuning path wins; otherwise fall back to
        // the env/table/heuristic ladder.  Both routes land here so there is
        // exactly one place that decides the launch geometry.
        int gemv_block_size = 0;
        int gemv_tile_n     = 0;
        const char* source  = "tuned";
        if(v.contains("block_size") and v.contains("tile_n"))
        {
            gemv_block_size = v.at("block_size").to<int>();
            gemv_tile_n     = v.at("tile_n").to<int>();
        }
        else
        {
            bool trusted = false;
            std::tie(gemv_block_size, gemv_tile_n) =
                get_gemv_config(N, K, ctx.get_current_device().get_gfx_name(), &trusted);
            source = trusted ? "table" : "heuristic";
        }

        // MIGRAPHX_INT4_GEMV_TRACE=1 prints the config actually used for every
        // shape, whatever chose it.  This is how you find out what the tuner
        // picked: the winning solution is what reaches compile_op.
        if(enabled(MIGRAPHX_INT4_GEMV_TRACE{}))
        {
            std::cout << "[int4_gemv] N=" << N << " K=" << K << " block_k=" << block_k
                      << " has_zp=" << (has_zp ? 1 : 0) << " -> block_size=" << gemv_block_size
                      << " tile_n=" << gemv_tile_n << " (" << source << ")" << std::endl;
        }

        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "int4_gemv_main";

        // Flat 1D grid: total_blocks = n_blocks * batch
        // Kernel recovers: n_block = blockIdx.x % n_blocks, batch_id = blockIdx.x / n_blocks
        auto batch     = inputs.front().lens().front();
        auto n_blocks  = (N + gemv_tile_n - 1) / gemv_tile_n;
        options.global = n_blocks * batch * gemv_block_size;
        options.local  = gemv_block_size;

        auto n_params = inputs.size();
        // Argument layout: p0=A, p1=B, p2=scales, then optionally zp, then optionally bias, then
        // output
        int next_arg = 3;

        std::string zp_cast;
        if(has_zp)
        {
            zp_cast = "auto* zp_ptr = reinterpret_cast<const migraphx::uint8_t*>(private_p" +
                      std::to_string(next_arg) + ");";
            next_arg++;
        }
        else
        {
            zp_cast = "const migraphx::uint8_t* zp_ptr = nullptr;";
        }

        std::string bias_cast;
        if(has_bias)
        {
            bias_cast = "auto* bias_ptr = reinterpret_cast<const _Float16*>(private_p" +
                        std::to_string(next_arg) + ");";
            next_arg++;
        }
        else
        {
            bias_cast = "const _Float16* bias_ptr = nullptr;";
        }

        std::string out_arg = "private_p" + std::to_string(next_arg);

        auto src = interpolate_string(int4_gemv_kernel_src,
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

    // Candidate grid for a shape we have not measured.
    //
    // BLOCK_SIZE {64,128,256} x TILE_N {2,4,8} = 9 points.  Chosen from the
    // Llama gate_up bracket (the only shape bracketed on both axes):
    // TILE_N=4 won at every BLOCK_SIZE tried, and BLOCK_SIZE was single-peaked
    // at 64 (32/64/128/256/512 = 164.32/229.29/226.50/219.85/204.55 tok/s
    // end-to-end).  The grid brackets that peak on both axes.
    //
    // Do NOT narrow this until the single peak is confirmed on a second shape.
    // BLOCK_SIZE has no fixed sign: phi3.5's 16384x3072 gate_up wants 256 and
    // Llama's 16384x2048 gate_up wants 64 -- same op, same arch, adjacent K,
    // opposite directions.
    //
    // TILE_N stops at 8 because 32/64 spill (732-1436 B scratch measured), and
    // 16 only appears under exhaustive tuning.
    static std::vector<value> candidate_configs(bool exhaustive)
    {
        std::vector<int> block_sizes = {64, 128, 256};
        std::vector<int> tile_ns     = {2, 4, 8};
        if(exhaustive)
        {
            block_sizes = {32, 64, 128, 256, 512};
            tile_ns     = {2, 4, 8, 16};
        }
        std::vector<value> result;
        for(auto bs : block_sizes)
            for(auto tn : tile_ns)
                result.push_back({{"block_size", bs}, {"tile_n", tn}});
        return result;
    }

    optional<tuning_config>
    get_tuning_config(context& ctx, instruction_ref ins, const operation& op, bool exhaustive) const
    {
        auto v = op.to_value();
        auto n = v.at("N").to<std::size_t>();
        auto k = v.at("K").to<std::size_t>();

        tuning_config tc;
        tc.problem               = to_value(to_shapes(ins->inputs()));
        tc.detailed_problem_info = "int4_gemv N=" + std::to_string(n) + " K=" + std::to_string(k);

        bool trusted = false;
        auto picked  = get_gemv_config(n, k, ctx.get_current_device().get_gfx_name(), &trusted);
        value seed{{"block_size", picked.first}, {"tile_n", picked.second}};

        // A shape with an env override or a measured[] row already has its
        // answer.  Returning exactly one solution makes compile_ops skip
        // benchmarking entirely and insert it straight into the problem cache,
        // so tuned models pay nothing for this hook being here.
        if(trusted and not exhaustive)
        {
            tc.solutions = {seed};
            return tc;
        }

        // Otherwise the config would have been a guess.  Benchmark instead.
        // The guess goes first so that when benchmarking is skipped
        // (MIGRAPHX_SKIP_BENCHMARKING, cross-compile) compile_ops takes
        // solutions.front() and behaviour is byte-for-byte what shipped before.
        tc.solutions = {seed};
        for(const auto& c : candidate_configs(exhaustive))
        {
            if(c != seed)
                tc.solutions.push_back(c);
        }
        return tc;
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto v = op.to_value();
        for(const auto& x : solution)
            v.insert(x);
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
