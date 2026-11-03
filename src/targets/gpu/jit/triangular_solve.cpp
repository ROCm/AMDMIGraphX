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
#include <migraphx/math.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const triangular_solve_kernel = R"__migraphx__(
#include <migraphx/kernels/triangular_solve.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void triangular_solve_kernel(void* a, void* b, void* x)
{
    make_tensors()(a, b, x)([](auto&&... xs) {
        triangular_solve<${lower}, ${unit_diagonal}, ${registers}>(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct triangular_solve_compiler : compiler<triangular_solve_compiler>
{
    std::vector<std::string> names() const { return {"triangular_solve"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        auto n            = out_s.lens().at(out_s.ndim() - 2);
        auto k            = out_s.lens().back();
        // Up to 64 rows the column of each lane stays in registers with A in
        // LDS, one workgroup per batch problem and the lanes over the columns;
        // beyond that one lane per (batch, column) works through memory
        const bool registers = n <= 64;
        if(registers)
        {
            auto problems = out_s.elements() / (n * k);
            auto local    = std::min<std::size_t>(1024, integer_divide_ceil(k, 64) * 64);
            options.set_launch_params(v, problems * local, local);
        }
        else
        {
            options.set_launch_params(v, compute_global_for(ctx, out_s.elements() / n));
        }
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = "triangular_solve_kernel";
        auto src               = interpolate_string(
            triangular_solve_kernel,
            {{"lower", v.get("lower", true) ? "true" : "false"},
             {"unit_diagonal", v.get("unit_diagonal", false) ? "true" : "false"},
             {"registers", registers ? "true" : "false"}});
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
