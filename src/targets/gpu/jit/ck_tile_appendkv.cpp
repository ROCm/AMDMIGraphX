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
#include <ck/host/device_fmha_appendkv/problem.hpp>
#include <ck/host/utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace appendkv = ck::host::device_fmha_appendkv;

// NOLINTNEXTLINE
static const char* const ck_tile_appendkv_kernel_src = R"__migraphx__(
#include <args.hpp>
#include <ck/host/device_fmha_appendkv/wrapper.hpp>
#include <migraphx/kernels/ck_appendkv.hpp>

using namespace migraphx;

extern "C" {
using KernelType = ${solution};

__launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void ${kernel}(${params})
{
    transform_args(make_tensors())(${args})([](auto... xs) {
        ck_appendkv<KernelType, ${rotary_dim}, ${has_mask}>(xs...);
    });
}

}

)__migraphx__";

struct ck_tile_appendkv_compiler : compiler<ck_tile_appendkv_compiler>
{
    std::vector<std::string> names() const { return {"ck_tile_appendkv"}; }

    // Undo a transpose so the innermost stride is 1.
    // The CK descriptor expects dims as [B, H, S, D] with 3 strides (batch, nhead, seq),
    // assuming the innermost hdim stride is always 1.
    // K_cache may be transposed for efficient attention (K^T), and V_cache may be
    // stored column-major. Swap the last two dims/strides to present the canonical view.
    static shape normalize_inner_shape(const shape& s)
    {
        if(s.strides().back() != 1)
        {
            auto l  = s.lens();
            auto st = s.strides();
            std::swap(l[2], l[3]);
            std::swap(st[2], st[3]);
            return {s.type(), l, st};
        }
        return s;
    }

    appendkv::Problem create_problem(const std::vector<shape>& inputs, const value& v) const
    {
        int rotary = v.at("rotary").to<int>();

        const auto& q_shape       = inputs[0];
        const auto& k_cache_shape = inputs[1];
        const auto& knew_shape    = inputs[2];
        const auto& v_cache_shape = inputs[3];

        appendkv::Problem prob;
        prob.batch          = q_shape.lens()[0];
        prob.nhead          = q_shape.lens()[1];
        prob.nhead_k        = k_cache_shape.lens()[1];
        prob.M              = q_shape.lens()[2];
        prob.N              = knew_shape.lens()[2];
        prob.K              = q_shape.lens()[3];
        prob.O              = v_cache_shape.lens()[3];
        prob.total_seqlen_k = k_cache_shape.lens()[2];
        prob.cache_seqlen   = prob.total_seqlen_k - prob.N;
        prob.dtype          = get_type(q_shape);
        prob.is_v_rowmajor  = (v_cache_shape.strides().back() == 1);

        if(rotary == 0)
            prob.rotary = appendkv::RotaryEmbeddingType::None;
        else if(rotary == 1)
            prob.rotary = appendkv::RotaryEmbeddingType::HalfRotated;
        else
            prob.rotary = appendkv::RotaryEmbeddingType::Interleaved;

        prob.rotary_dim = (rotary != 0) ? prob.K : 0;
        prob.has_mask   = (prob.M > 1);
        return prob;
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto tuning_value = v.get("tuning_value", 0);
        auto problem      = create_problem(inputs, v);
        auto arch         = ctx.get_current_device().get_gfx_name();


        auto solutions = problem.GetSolutions(arch);
        if(solutions.empty())
            MIGRAPHX_THROW("No AppendKV solutions for arch " + arch);
        const auto& solution    = solutions.at(tuning_value);
        const auto template_str = solution.ToTemplateString();

        auto m0 = solution.GetTemplateParameter<std::size_t>("M0");
        auto n0 = solution.GetTemplateParameter<std::size_t>("N0");
        std::size_t num_tiles = std::max(
            ck::host::integer_divide_ceil(problem.M, m0),
            ck::host::integer_divide_ceil(problem.N, n0));
        constexpr std::size_t block_size = 256;

        // Build virtual_inputs with normalized shapes for args.hpp generation.
        // Indices: 0=Q, 1=K_cache, 2=Knew, 3=V_cache, 4=Vnew, 5=seqlen_k, [6=cos, 7=sin]
        // K_cache (1) may be transposed for attention; V_cache (3) / Vnew (4) may be col-major.
        std::vector<shape> virtual_inputs;
        virtual_inputs.reserve(inputs.size());
        for(std::size_t i = 0; i < inputs.size(); ++i)
        {
            if(i >= 1 and i <= 4)
                virtual_inputs.push_back(normalize_inner_shape(inputs[i]));
            else
                virtual_inputs.push_back(inputs[i]);
        }

        hip_compile_options options;
        options.additional_src_files = ck_tile_headers();
        options.inputs               = inputs;
        options.virtual_inputs       = virtual_inputs;
        options.output               = inputs[1];
        options.output_arg           = 1;
        options.kernel_name          = v.get("kernel", std::string{"ck_tile_appendkv_kernel"});
        options.global               = num_tiles * block_size;
        options.global_y             = problem.nhead;
        options.global_z             = problem.batch;
        options.local                = block_size;
        options.emplace_param("-Wno-pass-failed");

        auto src =
            interpolate_string(ck_tile_appendkv_kernel_src,
                               {{"solution", template_str},
                                {"kernel", options.kernel_name},
                                {"params", enum_params(inputs.size(), "void * private_p")},
                                {"args", enum_params(inputs.size(), "private_p")},
                                {"rotary_dim", std::to_string(problem.rotary_dim)},
                                {"has_mask", (problem.M > 1) ? "true" : "false"}});

        return compile_hip_code_object(ctx, src, options);
    }

    value create_settings(instruction_ref, const operation& op) const
    {
        auto v      = op.to_value();
        v["kernel"] = "ck_tile_appendkv_kernel";
        return v;
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op, const value& solution) const
    {
        auto shapes = to_shapes(ins->inputs());
        auto v      = create_settings(ins, op);
        if(not solution.is_null())
            v["tuning_value"] = solution;
        auto co = compile_op(ctx, shapes, v);

        return {co, [](module& m, instruction_ref ins, const operation& compiled_op) {
            auto inputs = ins->inputs();

            auto kernel_ins = m.insert_instruction(ins, compiled_op, inputs);

            // Map tuple index -> original input buffer that the kernel modifies in-place
            // get_tuple_elem 0 -> q (inputs[0])
            // get_tuple_elem 1 -> k_cache (inputs[1])
            // get_tuple_elem 2 -> v_cache (inputs[3])
            std::array<instruction_ref, 3> buffers = {inputs[0], inputs[1], inputs[3]};

            auto outputs = ins->outputs();
            for(auto user : outputs)
            {
                if(user->name() != "get_tuple_elem")
                    continue;
                auto idx = user->get_operator().to_value().at("index").to<int>();
                m.replace_instruction(
                    user, make_op("identity"), buffers.at(idx), kernel_ins);
            }

            m.replace_instruction(ins, make_op("identity"), kernel_ins);
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
        std::vector<shape> key_shapes{shapes[0], shapes[1], shapes[2], shapes[3]};
        tc.problem = to_value(key_shapes);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
