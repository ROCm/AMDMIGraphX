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
#include <migraphx/op/common.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const rnn_var_sl_shift_sequence_kernel = R"__migraphx__(
#include <migraphx/kernels/rnn_variable_seq_lens.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void rnn_var_sl_shift_sequence_kernel(void* in_hs, void* in_sl, void* output) 
{
    make_tensors()(in_hs, in_sl, output)([](auto input, auto seq_lens, auto out) { 
        rnn_var_sl_shift_sequence(input, seq_lens, out); 
    });
}

}

} // namespace migraphx

)__migraphx__";

// NOLINTNEXTLINE
static const char* const rnn_var_sl_shift_output_kernel = R"__migraphx__(
#include <migraphx/kernels/rnn_variable_seq_lens.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void rnn_var_sl_shift_output_kernel(void* in_hs, void* in_sl, void* output) 
{
    make_tensors()(in_hs, in_sl, output)([](auto input, auto seq_lens, auto out) { 
        rnn_var_sl_shift_output<${is_reverse}>(input, seq_lens, out); 
    });
}

}

} // namespace migraphx

)__migraphx__";

// NOLINTNEXTLINE
static const char* const rnn_var_sl_last_output_kernel = R"__migraphx__(
#include <migraphx/kernels/rnn_variable_seq_lens.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void rnn_var_sl_last_output_kernel(void* in_hs, void* in_sl, void* output) 
{
    make_tensors()(in_hs, in_sl, output)([](auto input, auto seq_lens, auto out) { 
        rnn_var_sl_last_output<${is_reverse}>(input, seq_lens, out); 
    });
}

}

} // namespace migraphx

)__migraphx__";

struct rnn_var_sl_shift_sequence_compiler : compiler<rnn_var_sl_shift_sequence_compiler>
{
    std::vector<std::string> names() const { return {"rnn_var_sl_shift_sequence"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "rnn_var_sl_shift_sequence_kernel";
        options.virtual_inputs = inputs;

        const auto *src = rnn_var_sl_shift_sequence_kernel;

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

struct rnn_var_sl_shift_output_compiler : compiler<rnn_var_sl_shift_output_compiler>
{
    std::vector<std::string> names() const { return {"rnn_var_sl_shift_output"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "rnn_var_sl_shift_output_kernel";
        options.virtual_inputs = inputs;

        auto direction  = v.at("direction").to<op::rnn_direction>();
        auto is_reverse = (direction == op::rnn_direction::reverse);

        auto src = interpolate_string(rnn_var_sl_shift_output_kernel,
                                      {{"is_reverse", to_string(is_reverse)}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

struct rnn_var_sl_last_output_compiler : compiler<rnn_var_sl_last_output_compiler>
{
    std::vector<std::string> names() const { return {"rnn_var_sl_last_output"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "rnn_var_sl_last_output_kernel";
        options.virtual_inputs = inputs;

        auto direction  = v.get("direction", op::rnn_direction::forward);
        auto is_reverse = (direction == op::rnn_direction::reverse);

        auto src = interpolate_string(rnn_var_sl_last_output_kernel,
                                      {{"is_reverse", is_reverse ? "true" : "false"}});

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
