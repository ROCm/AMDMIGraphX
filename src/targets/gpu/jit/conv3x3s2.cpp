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
#include <migraphx/math.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const conv3x3s2_fp32_kernel = R"__migraphx__(
#include <migraphx/kernels/conv3x3s2.hpp>
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
        [](auto output, auto x, auto w, auto... inputs) {
            conv3x3s2_fp32<${tk}, ${tp}, ${nw}, float>(${post}, output, x, w, inputs...);
        });
}

}

} // namespace migraphx

)__migraphx__";

struct conv3x3s2_compiler : compiler<conv3x3s2_compiler>
{
    std::vector<std::string> names() const { return {"gpu::conv3x3s2", "conv3x3s2"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& x_s   = inputs.front();
        const auto& w_s   = inputs.at(1);
        const auto& out_s = inputs.back();
        if(x_s.type() != shape::float_type or x_s.lens().size() != 4 or w_s.lens().size() != 3)
            MIGRAPHX_THROW("conv3x3s2: expects an FP32 NCHW input and [C, 9, KPAD] weights");

        const auto tk = v.get("tk", std::size_t{8});
        const auto tp = v.get("tp", std::size_t{2});
        const auto nw = v.get("nw", std::size_t{4});
        if((tp != 1 and tp != 2 and tp != 4) or w_s.lens()[2] % tk != 0 or nw == 0 or nw > 8)
            MIGRAPHX_THROW("conv3x3s2: invalid tile");

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = v.get("kernel", std::string{"conv3x3s2_kernel"});

        // One workgroup per (image, block of TK output channels, run of 32 groups), NW waves
        // each. Example: output [1,256,15,20], TK=8, TP=2: groups = 15*10 = 150,
        // 5 runs of 32 x 32 channel blocks = 160 workgroups.
        const auto& out_lens = out_s.lens();
        const auto groups    = out_lens[2] * integer_divide_ceil(out_lens[3], tp);
        const auto num_blocks =
            out_lens[0] * integer_divide_ceil(out_lens[1], tk) * integer_divide_ceil(groups, 32);
        const auto block_size = nw * 32;
        options.set_launch_params(v, num_blocks * block_size, block_size);

        auto src = interpolate_string(conv3x3s2_fp32_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"tk", std::to_string(tk)},
                                       {"tp", std::to_string(tp)},
                                       {"nw", std::to_string(nw)},
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
            v["preamble"] = generate_pointwise(*pm, "post_conv3x3s2");
            v["post"]     = "MIGRAPHX_LIFT(post_conv3x3s2)";
            v["kernel"]   = "conv3x3s2_" + generate_name_from_ops(*pm) + "_kernel";
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }

    optional<tuning_config>
    get_tuning_config(const context&, instruction_ref ins, const operation&, bool) const
    {
        tuning_config tc;
        auto shapes = to_shapes(ins->inputs());
        tc.problem =
            value{{"version", 1},
                  {"shapes", to_value(shapes)},
                  {"post",
                   ins->module_inputs().empty()
                       ? std::string{}
                       : generate_pointwise(*ins->module_inputs().front(), "post_conv3x3s2")}};
        // Candidates: TK=8 with TP (output columns per lane) 1/2/4 and NW (waves splitting the
        // input channels) 1/2/4/8, plus TK=16, TP=2. Small outputs need more waves to fill the
        // GPU; large ones prefer bigger tiles. The fastest is picked by benchmarking.
        const auto kpad = shapes[1].lens()[2];
        for(std::size_t tp : {1, 2, 4})
            for(std::size_t nw : {1, 2, 4, 8})
                tc.solutions.push_back({{"tk", 8}, {"tp", tp}, {"nw", nw}});
        if(kpad % 16 == 0)
        {
            tc.solutions.push_back({{"tk", 16}, {"tp", 2}, {"nw", 1}});
            tc.solutions.push_back({{"tk", 16}, {"tp", 2}, {"nw", 4}});
        }
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
