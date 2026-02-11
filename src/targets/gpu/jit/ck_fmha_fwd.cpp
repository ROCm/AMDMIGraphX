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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/ck.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <ck/host/utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const ck_fmha_fwd_kernel = R"__migraphx__(
#include <args.hpp>
#include <${include}>
#include <migraphx/kernels/ck_fmha_fwd.hpp>

using namespace migraphx;

extern "C" {

__global__ void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto... xs) {
        ck_fmha_fwd<${solution}>(xs...);
    });
}

}

)__migraphx__";

struct ck_fmha_fwd_compiler : compiler<ck_fmha_fwd_compiler>
{
    std::vector<std::string> names() const { return {"ck_fmha_fwd", "gpu::ck_fmha_fwd"}; }

    ck::host::device_fmha_fwd::Problem create_problem(const std::vector<shape>& inputs,
                                                      const value&) const
    {
        // inputs: [Q, K, V, output] or [Q, K, bias, V, output]
        const auto& q_shape = inputs[0];
        const auto& k_shape = inputs[1];

        const bool has_bias = inputs.size() == 5;
        const auto& o_shape = inputs.back();

        auto rank = q_shape.ndim();

        ck::host::device_fmha_fwd::Problem prob;
        prob.batch = q_shape.lens()[rank - 4];
        prob.nhead = q_shape.lens()[rank - 3];
        prob.M     = q_shape.lens()[rank - 2]; // seqlen_q
        prob.N     = k_shape.lens()[rank - 2]; // seqlen_k
        prob.K     = q_shape.lens()[rank - 1]; // hdim_q
        prob.O     = o_shape.lens()[rank - 1]; // hdim_v
        prob.dtype = get_type(q_shape);

        const auto& v_shape = has_bias ? inputs[3] : inputs[2];
        prob.is_v_rowmajor  = (v_shape.strides().back() == 1);
        prob.is_causal      = false;
        prob.has_bias       = has_bias;

        return prob;
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto tuning_value = v.get("tuning_value", 0);
        auto problem      = create_problem(inputs, v);

        const auto& o_shape = inputs.back();

        auto arch = ctx.get_current_device().get_gfx_name();

        const auto include_header = problem.GetIncludeHeader();
        const auto solutions      = problem.GetSolutions(arch);
        if(solutions.empty())
            MIGRAPHX_THROW("No FMHA solutions for arch " + arch);
        const auto& solution    = solutions.at(tuning_value);
        const auto template_str = solution.ToTemplateString();

        // Compute launch dimensions
        auto bm0 = solution.GetTemplateParameter<std::size_t>("BM0");
        auto bn1 = solution.GetTemplateParameter<std::size_t>("BN1");

        auto rm0 = solution.GetTemplateParameter<std::size_t>("RM0");
        auto rn0 = solution.GetTemplateParameter<std::size_t>("RN0");
        auto rk0 = solution.GetTemplateParameter<std::size_t>("RK0");
        auto rm1 = solution.GetTemplateParameter<std::size_t>("RM1");
        auto rn1 = solution.GetTemplateParameter<std::size_t>("RN1");
        auto rk1 = solution.GetTemplateParameter<std::size_t>("RK1");

        const std::size_t warp_size  = ctx.get_current_device().get_wavefront_size();
        const std::size_t num_warps  = std::max(rm0 * rn0 * rk0, rm1 * rn1 * rk1);
        const std::size_t block_size = num_warps * warp_size;

        const std::size_t grid_m = ck::host::integer_divide_ceil(problem.M, bm0);
        const std::size_t grid_o = ck::host::integer_divide_ceil(problem.O, bn1);

        assert(v.contains("scale"));
        auto scale = v.at("scale").to<float>();

        hip_compile_options options;
        options.additional_src_files = ck_tile_headers();
        options.global               = problem.nhead * block_size;
        options.global_y             = grid_m * grid_o;
        options.global_z             = problem.batch;
        options.local                = block_size;
        options.local_y              = 1;
        options.local_z              = 1;
        options.inputs               = inputs;
        options.output               = o_shape;
        options.kernel_name          = v.get("kernel", std::string{"ck_fmha_fwd_kernel"});
        options.emplace_param("-DSCALE=" + std::to_string(scale));

        auto src = interpolate_string(ck_fmha_fwd_kernel,
                                      {{"include", include_header},
                                       {"solution", template_str},
                                       {"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")}});

        return compile_hip_code_object(ctx, src, options);
    }

    value create_settings(instruction_ref, const operation& op) const
    {
        auto v      = op.to_value();
        v["kernel"] = "ck_fmha_fwd_kernel";
        return v;
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto v      = create_settings(ins, op);
        if(not solution.is_null())
            v["tuning_value"] = solution;
        return {compile_op(ctx, shapes, v),
                [=](module& m, instruction_ref ins2, const operation& code_object) {
                    m.replace_instruction(ins2, code_object, ins2->inputs());
                }};
    }

    optional<tuning_config>
    get_tuning_config(context& ctx, instruction_ref ins, const operation& op, bool exhaustive) const
    {
        if(not exhaustive and not enabled(MIGRAPHX_TUNE_CK{}))
            return nullopt;
        tuning_config tc;
        auto shapes    = to_shapes(ins->inputs());
        auto problem   = create_problem(shapes, create_settings(ins, op));
        auto solutions = problem.GetSolutions(ctx.get_current_device().get_gfx_name());
        tc.solutions.resize(solutions.size());
        std::iota(tc.solutions.begin(), tc.solutions.end(), 0);
        std::vector<shape> key_shapes{shapes[0], shapes[1], shapes.back()};
        tc.problem = to_value(key_shapes);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
