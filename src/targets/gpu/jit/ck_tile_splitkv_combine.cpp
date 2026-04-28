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
#include <migraphx/shape.hpp>
#include <ck/host/device_fmha_splitkv_combine/problem.hpp>
#include <ck/host/utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

namespace combine = ck::host::device_fmha_splitkv_combine;

// NOLINTNEXTLINE
static const char* const ck_tile_splitkv_combine_kernel = R"__migraphx__(
#include <args.hpp>
#include <${include}>
#include <migraphx/kernels/ck_splitkv_combine.hpp>

using namespace migraphx;

extern "C" {
using KernelType = ${solution};

__launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last<1>())(${args})([](auto... xs) {
        ck_splitkv_combine<${solution}>(xs...);
    });
}

}

)__migraphx__";

struct ck_tile_splitkv_combine_compiler : compiler<ck_tile_splitkv_combine_compiler>
{
    std::vector<std::string> names() const { return {"gpu::ck_tile_splitkv_combine"}; }

    // Insert a dimension of size 1 at position 1 for 4D -> 5D normalization.
    // [B, S, M, O] -> [B, 1, S, M, O]
    static shape ensure_5d(const shape& s)
    {
        if(s.ndim() >= 5)
            return s;
        auto l  = s.lens();
        auto st = s.strides();
        l.insert(l.begin() + 1, 1);
        st.insert(st.begin() + 1, st[0]);
        return {s.type(), l, st};
    }

    // Insert a dimension of size 1 at position 1 for 3D -> 4D normalization.
    // [B, M, O] -> [B, 1, M, O]
    static shape ensure_4d(const shape& s)
    {
        if(s.ndim() >= 4)
            return s;
        auto l  = s.lens();
        auto st = s.strides();
        l.insert(l.begin() + 1, 1);
        st.insert(st.begin() + 1, st[0]);
        return {s.type(), l, st};
    }

    // Drop trailing dimension of size 1 from lse_acc.
    // [B, H, S, M, 1] -> [B, H, S, M]
    static shape drop_trailing_one(const shape& s)
    {
        auto l  = s.lens();
        auto st = s.strides();
        if(not l.empty() and l.back() == 1)
        {
            l.pop_back();
            st.pop_back();
        }
        return {s.type(), l, st};
    }

    combine::Problem create_problem(const std::vector<shape>& inputs, const value& v) const
    {
        // inputs: [o_acc, lse_acc, output]
        const auto& o_acc_shape = inputs[0];
        auto rank               = o_acc_shape.ndim();

        combine::Problem prob;
        if(rank == 5)
        {
            prob.batch = o_acc_shape.lens()[0];
            prob.nhead = o_acc_shape.lens()[1];
            prob.M     = o_acc_shape.lens()[3];
            prob.O     = o_acc_shape.lens()[4];
        }
        else
        {
            prob.batch = o_acc_shape.lens()[0];
            prob.nhead = 1;
            prob.M     = o_acc_shape.lens()[2];
            prob.O     = o_acc_shape.lens()[3];
        }
        prob.num_splits = v.at("num_splits").to<std::size_t>();
        prob.dtype      = get_type(inputs.back());
        return prob;
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto tuning_value = v.get("tuning_value", 0);
        auto problem      = create_problem(inputs, v);
        auto arch         = ctx.get_current_device().get_gfx_name();

        const auto include_header = problem.GetIncludeHeader();
        const auto solutions      = problem.GetSolutions(arch);
        if(solutions.empty())
            MIGRAPHX_THROW("No SplitKV Combine solutions for arch " + arch);
        const auto& solution    = solutions.at(tuning_value);
        const auto template_str = solution.ToTemplateString();

        auto kn1 = solution.GetTemplateParameter<std::size_t>("N1");
        auto km0 = solution.GetTemplateParameter<std::size_t>("M0");

        constexpr std::size_t block_size = 256;

        const std::size_t grid_x = ck::host::integer_divide_ceil(problem.M, km0) *
                                   ck::host::integer_divide_ceil(problem.O, kn1);
        const std::size_t grid_y = problem.nhead;
        const std::size_t grid_z = problem.batch;

        const auto& output_shape = inputs.back();

        // inputs: [o_acc, lse_acc, output]
        // o_acc is 5D or 4D, lse_acc is 5D or 4D (with trailing 1)
        auto flat_shapes           = flatten(inputs);
        const bool needs_nhead_dim = inputs[0].ndim() == 4;

        // flat_shapes: [o_acc, lse_acc, o]
        // Virtual shapes: normalize all to CK's expected ranks.
        // CK expects: o_acc 5D, lse_acc 4D (no trailing 1), o 4D
        std::vector<shape> virtual_shapes;
        virtual_shapes.reserve(flat_shapes.size());
        for(std::size_t i = 0; i < flat_shapes.size(); ++i)
        {
            if(i == 0)
            {
                // o_acc: ensure 5D [B, H, splits, M, O]
                virtual_shapes.push_back(needs_nhead_dim ? ensure_5d(flat_shapes[i])
                                                         : flat_shapes[i]);
            }
            else if(i == 1)
            {
                // lse_acc: drop trailing 1, then ensure 4D [B, H, splits, M]
                auto s = drop_trailing_one(flat_shapes[i]);
                virtual_shapes.push_back(needs_nhead_dim ? ensure_4d(s) : s);
            }
            else
            {
                // o (output): ensure 4D [B, H, M, O]
                virtual_shapes.push_back(needs_nhead_dim ? ensure_4d(flat_shapes[i])
                                                         : flat_shapes[i]);
            }
        }

        hip_compile_options options;
        options.additional_src_files = ck_tile_headers();
        options.inputs               = flat_shapes;
        options.virtual_inputs       = virtual_shapes;
        options.output               = output_shape;
        options.kernel_name = v.get("kernel", std::string{"ck_tile_splitkv_combine_kernel"});
        options.emplace_param("-DCK_TILE_FMHA_FWD_FAST_EXP2=1");
        options.emplace_param("-fgpu-flush-denormals-to-zero");
        options.emplace_param("-Wno-pass-failed");

        options.global   = grid_x * block_size;
        options.global_y = grid_y;
        options.global_z = grid_z;
        options.local    = block_size;
        options.local_y  = 1;
        options.local_z  = 1;

        auto src =
            interpolate_string(ck_tile_splitkv_combine_kernel,
                               {{"include", include_header},
                                {"solution", template_str},
                                {"kernel", options.kernel_name},
                                {"params", enum_params(flat_shapes.size(), "void * private_p")},
                                {"args", enum_params(flat_shapes.size(), "private_p")}});

        return compile_hip_code_object(ctx, src, options);
    }

    value create_settings(instruction_ref, const operation& op) const
    {
        auto v      = op.to_value();
        v["kernel"] = "ck_tile_splitkv_combine_kernel";
        return v;
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto v      = create_settings(ins, op);
        if(not solution.is_null())
            v["tuning_value"] = solution;
        return {compile_op(ctx, shapes, v)};
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
        std::vector<shape> key_shapes{shapes[0], shapes[1]};
        tc.problem = to_value(key_shapes);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
