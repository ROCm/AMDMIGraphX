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
#include <ck/host/device_fmha_splitkv/problem.hpp>
#include <ck/host/utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

namespace splitkv = ck::host::device_fmha_splitkv;

// NOLINTNEXTLINE
static const char* const ck_tile_splitkv_kernel = R"__migraphx__(
#include <args.hpp>
#include <${include}>
#include <migraphx/kernels/ck_splitkv.hpp>

using namespace migraphx;

extern "C" {

__global__ void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last<2>())(${args})([](auto... xs) {
        ck_splitkv<${solution}>(xs...);
    });
}

}

)__migraphx__";

struct ck_tile_splitkv_compiler : compiler<ck_tile_splitkv_compiler>
{
    std::vector<std::string> names() const { return {"gpu::ck_tile_splitkv"}; }

    // Undo a transpose on K so that the innermost stride is 1 (row-major hdim_q).
    // CK requires K in [B, nhead_k, N, K] layout with K (hdim_q) contiguous.
    // Currently only handles K that was transposed (last stride != 1).
    // TODO: Consider how to handle K created directly as [B, K_dim, N] (last stride == 1
    // but hdim_q is NOT the contiguous dimension). Options include inserting a transpose
    // in find_flash_decoding or requiring the pattern matcher to only match transposed K.
    static shape normalize_k_shape(const shape& k)
    {
        if(k.strides().back() != 1)
        {
            auto rank = k.ndim();
            auto l    = k.lens();
            auto s    = k.strides();
            std::swap(l[rank - 2], l[rank - 1]);
            std::swap(s[rank - 2], s[rank - 1]);
            return {k.type(), l, s};
        }
        return k;
    }

    // Insert a dimension of size 1 at position 1 for 3D -> 4D normalization.
    // [B, M, K] -> [B, 1, M, K]
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

    splitkv::Problem create_problem(const std::vector<shape>& inputs, const value& v) const
    {
        const auto& q_shape = inputs[0];
        const auto& k_shape = inputs[1];
        const auto& v_shape = inputs[2];
        auto rank           = q_shape.ndim();

        splitkv::Problem prob;
        if(rank == 4)
        {
            prob.batch   = q_shape.lens()[0];
            prob.nhead   = q_shape.lens()[1];
            prob.nhead_k = k_shape.lens()[1];
        }
        else
        {
            prob.batch   = q_shape.lens()[0];
            prob.nhead   = 1;
            prob.nhead_k = 1;
        }
        prob.M             = q_shape.lens()[rank - 2];
        prob.N             = k_shape.lens()[rank - 1];
        prob.K             = q_shape.lens()[rank - 1];
        prob.O             = v_shape.lens()[rank - 1];
        prob.num_splits    = v.at("num_splits").to<std::size_t>();
        prob.dtype         = get_type(q_shape);
        prob.o_acc_dtype   = get_type(inputs.back().sub_shapes()[0]);
        prob.is_v_rowmajor = (v_shape.strides().back() == 1);
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
            MIGRAPHX_THROW("No SplitKV solutions for arch " + arch);
        const auto& solution    = solutions.at(tuning_value);
        const auto template_str = solution.ToTemplateString();

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

        const bool merge_qk =
            solution.GetTemplateParameter<std::string>("MergeNumHeadGroupsSeqLenQ") == "true";
        const std::size_t m_eff =
            merge_qk ? problem.M * (problem.nhead / problem.nhead_k) : problem.M;
        const std::size_t nhead_eff = merge_qk ? problem.nhead_k : problem.nhead;

        const std::size_t grid_x =
            ck::host::integer_divide_ceil(m_eff, bm0) *
            ck::host::integer_divide_ceil(problem.O, bn1) * problem.num_splits;
        const std::size_t grid_y = nhead_eff;
        const std::size_t grid_z = problem.batch;

        float scale = 1.0f;

        // Build the output tuple shape from the op's compute_shape.
        // inputs to this function: [Q, K, V, tuple(o_acc, lse_acc)]
        // The last element is the pre-allocated tuple output.
        const auto& output_shape = inputs.back();

        // Flatten all shapes: [Q, K, V, o_acc, lse_acc]
        auto flat_shapes = flatten(inputs);
        const bool needs_nhead_dim = inputs[0].ndim() == 3;

        std::vector<shape> virtual_shapes;
        virtual_shapes.reserve(flat_shapes.size());
        for(std::size_t i = 0; i < flat_shapes.size(); ++i)
        {
            if(i == 1)
            {
                auto s = normalize_k_shape(flat_shapes[i]);
                virtual_shapes.push_back(needs_nhead_dim ? ensure_4d(s) : s);
            }
            else if(i < 3)
            {
                virtual_shapes.push_back(needs_nhead_dim ? ensure_4d(flat_shapes[i])
                                                         : flat_shapes[i]);
            }
            else if(i == 3)
            {
                virtual_shapes.push_back(needs_nhead_dim ? ensure_5d(flat_shapes[i])
                                                         : flat_shapes[i]);
            }
            else
            {
                auto s = drop_trailing_one(flat_shapes[i]);
                virtual_shapes.push_back(needs_nhead_dim ? ensure_5d(s) : s);
            }
        }


        hip_compile_options options;
        options.additional_src_files = ck_tile_headers();
        options.inputs               = flat_shapes;
        options.virtual_inputs       = virtual_shapes;
        options.output               = output_shape;
        options.kernel_name = v.get("kernel", std::string{"ck_tile_splitkv_kernel"});
        options.emplace_param("-DMIGRAPHX_CK_SPLITKV_SCALE=" + std::to_string(scale));

        options.global   = grid_x * block_size;
        options.global_y = grid_y;
        options.global_z = grid_z;
        options.local    = block_size;
        options.local_y  = 1;
        options.local_z  = 1;

        auto src =
            interpolate_string(ck_tile_splitkv_kernel,
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
        v["kernel"] = "ck_tile_splitkv_kernel";
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
        std::vector<shape> key_shapes{shapes[0], shapes[1], shapes[2]};
        tc.problem = to_value(key_shapes);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
