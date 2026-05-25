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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

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
            winograd_conv_f23_wmma<${nw}, ${cb}, ${kw}, ${sk}>(
                ${post}, output, x, u, inputs...);
        });
}

}

} // namespace migraphx

)__migraphx__";

struct winograd_conv_compiler : compiler<winograd_conv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::winograd_conv", "winograd_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s      = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = v.get("kernel", std::string{"winograd_conv_kernel"});

        const auto nw                = v.get("nw", std::size_t{4});
        const auto cb                = v.get("cb", std::size_t{16});
        const auto kw                = v.get("kw", std::size_t{1});
        // sk = within-WG c-axis split factor. sk=1 is the original behavior;
        // sk>1 has nw/sk NT-groups per workgroup with sk waves cooperating on
        // the c contraction (cross-wave LDS reduce at the end). When sk>1, kw
        // is forced to 1 (LDS budget for per-wave U slots would otherwise
        // overflow).
        const auto sk                = v.get("sk", std::size_t{1});
        const std::size_t bk         = 16;
        const std::size_t bk_wg      = bk * kw;
        // BT = BT_per_wave * (NW/SK). SK splits waves within a workgroup
        // across the c contraction, so each WG covers fewer NT tiles per round
        // when SK>1, increasing total WG count.
        const std::size_t nt_groups  = nw / sk;
        const std::size_t bt         = 16 * nt_groups;
        const std::size_t block_size = nw * 32;

        const auto& out_lens = out_s.lens();
        const auto N         = out_lens[0];
        const auto K         = out_lens[1];
        const auto H_out     = out_lens[2];
        const auto W_out     = out_lens[3];
        const auto tiles_h   = (H_out + 1) / 2;
        const auto tiles_w   = (W_out + 1) / 2;
        const auto NT_total  = N * tiles_h * tiles_w;

        const auto k_wg_blocks = (K + bk_wg - 1) / bk_wg;
        const auto t_blocks    = (NT_total + bt - 1) / bt;
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
                                       {"post", v.get("post", std::string{"op::id{}"})},
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
            auto* pm      = ins->module_inputs().front();
            v["preamble"] = generate_pointwise(*pm, "post_winograd_conv");
            v["post"]     = "MIGRAPHX_LIFT(post_winograd_conv)";
            v["kernel"]   = "winograd_conv_" + generate_name_from_ops(*pm) + "_kernel";
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config>
    get_tuning_config(const context&, instruction_ref ins, const operation&, bool) const
    {
        tuning_config tc;
        auto shapes = to_shapes(ins->inputs());
        tc.problem  = to_value(shapes);

        // Wave32 WMMA configs. CB must be a multiple of WMMA K (16). KW is
        // the number of K_blocks (BK=16 each) processed per workgroup.
        // V values now live in per-lane registers (no V_lds), so LDS budget
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
