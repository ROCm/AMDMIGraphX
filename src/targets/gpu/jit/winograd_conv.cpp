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
#include <cassert>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// The fused pointwise's first parameter is the winograd conv result. If the
// pointwise's first op converts it to a wider type (e.g. fp32), the post-op
// genuinely computes at that wider precision, so the kernel should feed `f`
// the fp32 accumulator directly rather than rounding it to the conv's half
// precision first. Otherwise the conv result is cast to its natural type
// (half) so the post-op matches the unfused conv + pointwise reference. Return
// the C++ type the kernel should cast the conv result to before calling `f`.
static std::string post_input_cast(const module& pm)
{
    // Pointwise submodule params are named x0, x1, ...; x0 is arg 0, which the
    // fusion wires to the winograd conv output. Its type is the conv's natural
    // output precision (half for the fp16 kernel, float for the fp32 kernel),
    // which is the base type the post-op computes at.
    auto x0 = pm.get_parameter("x0");
    if(x0 == pm.end())
        return "half";
    const std::string base = shape::cpp_type(x0->get_shape().type());
    // Only treat a *leading* convert as the post-op's compute type, i.e. when
    // the conv result feeds exactly one op and that op is a convert to a type
    // wider than the conv output. A convert that appears later (after an
    // add/activation/etc.) must still run at conv precision first.
    const auto& users = x0->outputs();
    if(users.size() != 1)
        return base;
    auto user = users.front();
    if(user->name() != "convert")
        return base;
    auto t = user->get_shape().type();
    if(shape{t}.type_size() <= x0->get_shape().type_size())
        return base;
    return shape::cpp_type(t);
}

// NOLINTNEXTLINE
static const char* const winograd_conv_kernel = R"__migraphx__(
#include <migraphx/kernels/winograd_conv.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})(
        [](auto output, auto x, auto u, auto... inputs) {
            winograd_conv_f23_wmma<${nw}, ${cb}, ${kw}, ${sk}, ${ft}, ${nhwc}, ${conv_cast}>(
                ${post}, output, x, u, inputs...);
        });
}

}

} // namespace migraphx

)__migraphx__";

// fp32 FMA/DPP kernel (winograd_conv_f23_fp32). Configured by nw (waves),
// ko (output channels per lane) and tiles (winograd tiles per quad).
// NOLINTNEXTLINE
static const char* const winograd_conv_fp32_kernel = R"__migraphx__(
#include <migraphx/kernels/winograd_conv_fp32.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})(
        [](auto output, auto x, auto u, auto... inputs) {
            winograd_conv_f23_fp32<${nw}, ${ko}, ${tiles}, ${sk}, ${pipe}, ${cu}, ${sstore}, ${conv_cast}>(
                ${post}, output, x, u, inputs...);
        });
}

}

} // namespace migraphx

)__migraphx__";

struct winograd_conv_compiler : compiler<winograd_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::winograd_conv", "winograd_conv"}; }

    // fp32 FMA/DPP kernel: lane%4 = winograd v-column, quad = TILES tiles,
    // each wave = 8 quads. Launch covers (tile groups) x (output-channel blocks),
    // where a tile group is one workgroup's worth of quads.
    operation compile_op_fp32(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s      = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = v.get("kernel", std::string{"winograd_conv_fp32_kernel"});

        const auto nw    = v.get("nw", std::size_t{4});
        const auto ko    = v.get("ko", std::size_t{8});
        const auto tiles = v.get("tiles", std::size_t{2});
        // sk = within-WG channel-split factor (must divide nw). sk>1 has nw/sk
        // NT-groups whose sk waves split the channel contraction.
        const auto sk = v.get("sk", std::size_t{1});
        // pipe = software-pipeline the input transform into the FMA loop (true) vs
        // the simple transform-then-FMA path (false). pipe costs a second live
        // v_reg, so the tuner only offers it on small (non-spilling) solutions.
        const bool pipe = v.get("pipe", false);
        // cu = channel-unroll (weight load width): 4=b128, 2=b64, 1=b32. Smaller
        // cu shrinks the pipelined double-buffer (finer-grained pipeline) at the
        // cost of more, narrower weight loads.
        const auto cu = v.get("cu", std::size_t{4});
        // S-store: the winograd weight literal is the v-half-transformed S=[3,4,K,C]
        // (dim0==3) rather than the full U=[4,4,K,C] (dim0==4); the kernel finishes
        // the register-only G u-transform, trading 3/4 the weight loads+bytes for a
        // few register FMAs (for weight-bandwidth-bound shapes).
        const bool sstore = inputs.at(1).lens().at(0) == 3;

        // Only nw/sk NT-groups' worth of distinct tiles are covered per WG.
        const std::size_t quads_per_wg = 8 * (nw / sk);
        const std::size_t block_size   = nw * 32;

        const auto& out_lens = out_s.lens();
        assert(out_lens.size() == 4);
        const auto n        = out_lens[0];
        const auto out_c    = out_lens[1];
        const auto out_h    = out_lens[2];
        const auto out_w    = out_lens[3];
        const auto tiles_h  = (out_h + 1) / 2;
        const auto tiles_w  = (out_w + 1) / 2;
        const auto nt_total = n * tiles_h * tiles_w;

        // One workgroup per (tile_group, k_block); its nw/sk NT-groups cover a
        // contiguous run of tiles for that k_block.
        const auto k_blocks    = (out_c + ko - 1) / ko;
        const auto quad_groups = (nt_total + tiles - 1) / tiles;
        const auto tile_blocks = (quad_groups + quads_per_wg - 1) / quads_per_wg;
        const auto num_blocks  = k_blocks * tile_blocks;

        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto src = interpolate_string(winograd_conv_fp32_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"nw", std::to_string(nw)},
                                       {"ko", std::to_string(ko)},
                                       {"tiles", std::to_string(tiles)},
                                       {"sk", std::to_string(sk)},
                                       {"pipe", pipe ? "true" : "false"},
                                       {"cu", std::to_string(cu)},
                                       {"sstore", sstore ? "true" : "false"},
                                       {"post", v.get("post", std::string{"op::id{}"})},
                                       {"conv_cast", v.get("conv_cast", std::string{"float"})},
                                       {"preamble", v.get("preamble", std::string{})}});

        return compile_hip_code_object(ctx, src, options);
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.front().type() == shape::float_type)
            return compile_op_fp32(ctx, inputs, v);
        hip_compile_options options;
        const auto& out_s      = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = v.get("kernel", std::string{"winograd_conv_kernel"});

        const auto nw = v.get("nw", std::size_t{4});
        const auto cb = v.get("cb", std::size_t{16});
        const auto kw = v.get("kw", std::size_t{1});
        // sk = within-WG c-axis split factor. sk=1 is the original behavior;
        // sk>1 has nw/sk NT-groups per workgroup with sk waves cooperating on
        // the c contraction (cross-wave LDS reduce at the end). When sk>1, kw
        // is forced to 1 (LDS budget for per-wave U slots would otherwise
        // overflow).
        const auto sk           = v.get("sk", std::size_t{1});
        const std::size_t bk    = 16;
        const std::size_t bk_wg = bk * kw;
        // BT = BT_per_wave * (NW/SK). SK splits waves within a workgroup
        // across the c contraction, so each WG covers fewer NT tiles per round
        // when SK>1, increasing total WG count.
        const std::size_t nt_groups  = nw / sk;
        const std::size_t bt         = 16 * nt_groups;
        const std::size_t block_size = nw * 32;

        // NHWC input: the convolution input (first operand) has its channel
        // axis innermost (stride 1). The kernel then reads channels coalesced.
        const auto& x_shape = inputs.front();
        const bool nhwc     = x_shape.lens().size() == 4 and x_shape.strides()[1] == 1;

        const auto& out_lens = out_s.lens();
        assert(out_lens.size() == 4);
        const auto n        = out_lens[0];
        const auto out_c    = out_lens[1];
        const auto out_h    = out_lens[2];
        const auto out_w    = out_lens[3];
        const auto tiles_h  = (out_h + 1) / 2;
        const auto tiles_w  = (out_w + 1) / 2;
        const auto nt_total = n * tiles_h * tiles_w;

        const auto k_wg_blocks = (out_c + bk_wg - 1) / bk_wg;
        const auto t_blocks    = (nt_total + bt - 1) / bt;
        const auto num_blocks  = k_wg_blocks * t_blocks;

        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto src = interpolate_string(winograd_conv_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"nw", std::to_string(nw)},
                                       {"cb", std::to_string(cb)},
                                       {"kw", std::to_string(kw)},
                                       {"sk", std::to_string(sk)},
                                       {"ft", v.get("full_transform", false) ? "true" : "false"},
                                       {"nhwc", nhwc ? "true" : "false"},
                                       {"post", v.get("post", std::string{"op::id{}"})},
                                       {"conv_cast", v.get("conv_cast", std::string{"half"})},
                                       {"preamble", v.get("preamble", std::string{})}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto v = op.to_value();
        for(const auto& s : solution)
            v.insert(s);
        if(not ins->module_inputs().empty())
        {
            auto* pm       = ins->module_inputs().front();
            v["preamble"]  = generate_pointwise(*pm, "post_winograd_conv");
            v["post"]      = "MIGRAPHX_LIFT(post_winograd_conv)";
            v["kernel"]    = "winograd_conv_" + generate_name_from_ops(*pm) + "_kernel";
            v["conv_cast"] = post_input_cast(*pm);
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config>
    get_tuning_config(const context&, instruction_ref ins, const operation&, bool) const
    {
        tuning_config tc;
        auto shapes = to_shapes(ins->inputs());
        tc.problem  = to_value(shapes);

        // fp32 FMA/DPP configs: nw (waves), ko (out-channels/lane), tiles
        // (winograd tiles/quad). ko*tiles is kept in ~16-32 (accumulators/lane =
        // 4*ko*tiles = 64-128) to bound register spilling.
        if(shapes.front().type() == shape::float_type)
        {
            // Larger ko amortizes the (out-channel-independent) input transform
            // over more output channels -- important when out_c is large; larger
            // tiles amortizes the shared weight load -- good for large in_c and
            // small out_c.
            tc.solutions.push_back({{"nw", 2}, {"ko", 8}, {"tiles", 2}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 2}});
            tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 2}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 1}});
            tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 1}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 4}});
            tc.solutions.push_back({{"nw", 2}, {"ko", 16}, {"tiles", 1}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 16}, {"tiles", 1}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 16}, {"tiles", 2}});
            tc.solutions.push_back({{"nw", 2}, {"ko", 16}, {"tiles", 2}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 32}, {"tiles", 1}});
            // pipe=1: software-pipeline the input transform into the FMA loop so
            // the DPP transform ops interleave with the FMAs instead of clustering
            // ahead (helps FMA-throughput-bound shapes -- large out_c x spatial).
            // The pipeline needs a second live v_reg, which spills for tiles>=4 or
            // ko>=16, so only the small ko=8/tiles<=2 solutions are offered; the
            // tuner keeps pipe=1 only where it beats the simple path.
            tc.solutions.push_back({{"nw", 2}, {"ko", 8}, {"tiles", 2}, {"pipe", true}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 2}, {"pipe", true}});
            tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 2}, {"pipe", true}});
            tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 1}, {"pipe", true}});
            tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 1}, {"pipe", true}});
            // Finer-grained pipeline: cu=2 (b64 weight load) halves the pipelined
            // double-buffer, so ko can go to 16 without spilling -- amortizing the
            // (non-dual-issue) DPP transforms over more FMAs. Costs more, narrower
            // weight loads; the tuner keeps it only where the trade pays off.
            tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 2}, {"pipe", true}, {"cu", 2}});
            tc.solutions.push_back(
                {{"nw", 4}, {"ko", 16}, {"tiles", 1}, {"pipe", true}, {"cu", 2}});
            tc.solutions.push_back(
                {{"nw", 2}, {"ko", 16}, {"tiles", 2}, {"pipe", true}, {"cu", 2}});
            tc.solutions.push_back(
                {{"nw", 4}, {"ko", 16}, {"tiles", 2}, {"pipe", true}, {"cu", 2}});
            // Channel-split (sk>1): nw/sk NT-groups whose sk waves split the
            // channel contraction and reduce partial M through LDS. Helps shapes
            // with few tiles + many channels (small spatial, large in_c/out_c),
            // where the plain path leaves waves idle. LDS = nw*32*4*tiles*ko
            // floats, so keep tiles=1, ko<=8. Only OFFERED when tiles are scarce
            // (small nt_total): on tile-rich shapes the plain path already fills
            // the machine, and offering sk configs there only adds tuner noise.
            const auto& out_lens = shapes.back().lens();
            const auto nt_total  = out_lens[0] * ((out_lens[2] + 1) / 2) * ((out_lens[3] + 1) / 2);
            if(nt_total < 256)
            {
                tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 1}, {"sk", 2}});
                tc.solutions.push_back({{"nw", 4}, {"ko", 8}, {"tiles", 1}, {"sk", 4}});
                tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 1}, {"sk", 2}});
                tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 1}, {"sk", 4}});
                tc.solutions.push_back({{"nw", 8}, {"ko", 8}, {"tiles", 1}, {"sk", 8}});
            }
            return tc;
        }

        // Wave32 WMMA configs. CB must be a multiple of WMMA K (16). KW is
        // the number of K_blocks (BK=16 each) processed per workgroup.
        // V values live in per-lane registers, so LDS budget
        // is just U_lds = KW * 16 * 16 * CB * 2 bytes (8KB per KW=1).
        // KW=1 is usually optimal because V is already free per-lane; KW>1
        // only helps to share U across more K outputs (rarely a win).
        // sk=1: original (no c-split) configs.
        tc.solutions.push_back({{"nw", 1}, {"cb", 16}, {"kw", 1}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 2}, {"cb", 16}, {"kw", 1}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 4}, {"cb", 16}, {"kw", 1}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 6}, {"cb", 16}, {"kw", 1}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 8}, {"cb", 16}, {"kw", 1}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 2}, {"cb", 32}, {"kw", 1}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 1}, {"cb", 16}, {"kw", 2}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 2}, {"cb", 16}, {"kw", 2}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 4}, {"cb", 16}, {"kw", 2}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 6}, {"cb", 16}, {"kw", 2}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 1}, {"cb", 16}, {"kw", 4}, {"sk", 1}});
        tc.solutions.push_back({{"nw", 4}, {"cb", 16}, {"kw", 3}, {"sk", 1}});
        // sk>1: within-WG c-axis split. KW must be 1. Helpful for shapes
        // where total WG count is limited (small NT or single K_block) — sk>1
        // increases NT-groups-per-WG counts and partitions the c contraction
        // across cooperating waves with an LDS cross-wave reduce.
        // LDS budget caps NW*SK to ~NW=4 SK=4 (48KB) — NW>=6 + SK>=2 overflows
        // due to per-wave U slots (NW*8KB) + y_reduce (NW*4KB).
        tc.solutions.push_back({{"nw", 2}, {"cb", 16}, {"kw", 1}, {"sk", 2}});
        tc.solutions.push_back({{"nw", 4}, {"cb", 16}, {"kw", 1}, {"sk", 2}});
        tc.solutions.push_back({{"nw", 4}, {"cb", 16}, {"kw", 1}, {"sk", 4}});
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
