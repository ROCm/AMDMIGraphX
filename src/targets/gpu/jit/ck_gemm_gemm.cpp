/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/filesystem.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/ck.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <ck/host/utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const ck_gemm_gemm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/ck_gemm_gemm.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/ops.hpp>
#include <${include}>

namespace migraphx {

${preamble0}

${preamble1}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto... xs) {
        ck_gemm_gemm<${solution}, ${blocks_per_batch}, ${d0s_count}>(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct ck_gemm_gemm_compiler : compiler<ck_gemm_gemm_compiler>
{
    std::vector<std::string> names() const { return {"ck_gemm_gemm", "gpu::ck_gemm_gemm"}; }

    ck::host::device_batched_gemm_multiple_d_gemm_multiple_d::Problem
    create_problem(const std::vector<shape>& inputs, const value& v) const
    {
        const auto& a0_shape = inputs[0];
        const auto& b0_shape = inputs[1];
        const auto& b1_shape = inputs[2];
        const auto& e1_shape = inputs.back();

        // cppcheck-suppress unreadVariable
        auto rank        = a0_shape.ndim();
        auto batch_count = get_batch_count(e1_shape);
        auto m           = e1_shape.lens()[rank - 2];
        m                = can_fold_batch(inputs) ? m * batch_count : m;
        auto n           = b0_shape.lens().back();
        auto k           = a0_shape.lens().back();
        auto o           = e1_shape.lens().back();

        const bool trans_a0 = transposed_matrix(a0_shape);
        const bool trans_b0 = transposed_matrix(b0_shape);
        const bool trans_b1 = transposed_matrix(b1_shape);
        const bool trans_e1 = transposed_matrix(e1_shape);

        auto d0s_count = v.get("d0s_count", 0);
        auto d1s_count = v.get("d1s_count", 0);

        std::vector<bool> trans_d0s;
        std::transform(inputs.begin() + 3,
                       inputs.end() - 1 - d1s_count,
                       std::back_inserter(trans_d0s),
                       [](const auto& i) { return transposed_matrix(i); });

        std::vector<bool> trans_d1s;
        std::transform(inputs.begin() + 3 + d0s_count,
                       inputs.end() - 1,
                       std::back_inserter(trans_d1s),
                       [](const auto& i) { return transposed_matrix(i); });

        const auto a0_type = get_type(a0_shape);
        const auto b0_type = get_type(b0_shape);
        const auto b1_type = get_type(b1_shape);
        const auto e1_type = get_type(e1_shape);

        std::vector<ck::host::DataType> d0s_type;
        std::transform(inputs.begin() + 3,
                       inputs.end() - 1 - d1s_count,
                       std::back_inserter(d0s_type),
                       [](const auto& i) { return get_type(i); });

        std::vector<ck::host::DataType> d1s_type;
        std::transform(inputs.begin() + 3 + d0s_count,
                       inputs.end() - 1,
                       std::back_inserter(d1s_type),
                       [](const auto& i) { return get_type(i); });

        std::string ck_passthrough = "ck_passthrough";
        std::string cde0_op        = ck_passthrough;
        std::string cde1_op        = ck_passthrough;
        assert(inputs.size() < 5 or v.contains("cde0_op") or v.containse("cde1_op"));
        if(v.contains("cde0_op"))
        {
            cde0_op = v.at("cde0_op").to<std::string>();
        }
        if(v.contains("cde1_op"))
        {
            cde1_op = v.at("cde1_op").to<std::string>();
        }

        return ck::host::device_batched_gemm_multiple_d_gemm_multiple_d::Problem{m,
                                                                                 n,
                                                                                 k,
                                                                                 o,
                                                                                 trans_a0,
                                                                                 trans_b0,
                                                                                 trans_d0s,
                                                                                 trans_b1,
                                                                                 trans_d1s,
                                                                                 trans_e1,
                                                                                 a0_type,
                                                                                 b0_type,
                                                                                 d0s_type,
                                                                                 b1_type,
                                                                                 d1s_type,
                                                                                 e1_type,
                                                                                 ck_passthrough,
                                                                                 ck_passthrough,
                                                                                 cde0_op,
                                                                                 ck_passthrough,
                                                                                 cde1_op};
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        std::cout << "compiling ck_gemm_gemm: " << v.get("kernel", std::string{}) << std::endl;
        
        const auto& e1_shape = inputs.back();
        auto tuning_value    = v.get("tuning_value", 0);
        auto batch_count     = get_batch_count(e1_shape);
        auto problem         = create_problem(inputs, v);

        const auto include_header = problem.GetIncludeHeader();
        const auto solutions =
            problem.GetSolutions(ctx.get_current_device().get_gfx_name(), "", "");
        const auto& solution        = solutions.at(tuning_value);
        const auto template_str     = solution.ToTemplateString();
        const auto block_size       = solution.GetTemplateParameter<std::size_t>("BlockSize");
        const auto m_per_block      = solution.GetTemplateParameter<std::size_t>("Gemm0MPerBlock");
        const auto n1_per_block     = solution.GetTemplateParameter<std::size_t>("Gemm1NPerBlock");
        const auto blocks_per_batch = ck::host::integer_divide_ceil(problem.M, m_per_block) *
                                      ck::host::integer_divide_ceil(problem.O, n1_per_block);

        hip_compile_options options;
        options.additional_src_files = ck_headers();

        auto grid_size = can_fold_batch(inputs) ? blocks_per_batch : batch_count * blocks_per_batch;
        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = e1_shape;
        options.kernel_name    = v.get("kernel", "ck_gemm_gemm_kernel");
        options.virtual_inputs = inputs;
        if(can_fold_batch(inputs))
        {
            auto vinputs = inputs;
            fold_batch_dims(vinputs[0]);
            remove_batch_dims(vinputs[1]);
            std::for_each(vinputs.begin() + 2, vinputs.end(), fold_batch_dims);
            options.virtual_inputs = vinputs;
        }

        if(v.get("check", false) or enabled(MIGRAPHX_CK_DEBUG{}))
            options.emplace_param("-DMIGRAPHX_CK_CHECK=1");

        auto src = interpolate_string(ck_gemm_gemm_kernel,
                                      {{"solution", template_str},
                                       {"include", include_header},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"blocks_per_batch", to_string(blocks_per_batch)},
                                       {"d0s_count", to_string(v.get("d0s_count", 0))},
                                       {"preamble0", v.get("preamble0", std::string{})},
                                       {"preamble1", v.get("preamble1", std::string{})},
                                       {"kernel", options.kernel_name}});

        return compile_hip_code_object(src, options);
    }

    value create_settings(instruction_ref ins, const operation& op) const
    {
        auto v = op.to_value();

        auto d0s_count = v.get("d0s_count", 0);
        auto d1s_count = v.get("d1s_count", 0);

        std::string pw0_name, pw1_name, kernel_name = "ck_gemm_";
        if(d0s_count > 0)
        {
            auto* pw0m     = ins->module_inputs().front();
            v["preamble0"] = generate_pointwise(*pw0m, "post_ck_gemm0_function") +
                             "\nMIGRAPHX_LIFT_CLASS(post_ck_gemm0, post_ck_gemm0_function);";
            v["cde0_op"] = "ck_function_adaptor<post_ck_gemm0>";
            kernel_name += generate_name_from_ops(*pw0m) + "_";
        }
        kernel_name += "gemm_";
        if(d1s_count > 0)
        {
            auto* pw1m     = ins->module_inputs().back();
            v["preamble1"] = generate_pointwise(*pw1m, "post_ck_gemm1_function") +
                             "\nMIGRAPHX_LIFT_CLASS(post_ck_gemm1, post_ck_gemm1_function);";
            v["cde1_op"] = "ck_function_adaptor<post_ck_gemm1>";
            kernel_name += generate_name_from_ops(*pw1m) + "_";
        }

        v["kernel"] = to_c_id(kernel_name + "kernel");
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
                    if(enabled(MIGRAPHX_LOG_CK_GEMM{}))
                    {
                        std::vector<shape> gemm_shapes{
                            shapes[0], shapes[1], shapes.back().with_type(shapes[0].type())};
                        std::cout << "gpu::ck_gemm_gemm: " << to_json_string(to_value(gemm_shapes))
                                  << std::endl;
                    }
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
        auto solutions = problem.GetSolutions(ctx.get_current_device().get_gfx_name(), "", "");
        tc.solutions.resize(solutions.size());
        std::iota(tc.solutions.begin(), tc.solutions.end(), 0);
        std::vector<shape> gemm_shapes{shapes[0], shapes[1], shapes.back()};
        tc.problem = to_value(gemm_shapes);
        return tc;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
