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
#include "verify.hpp"

#include <migraphx/compile_options.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program_verify.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Gives tolerances based on user input (`rms_tol`, `atol`, `rtol` parameters) and defaults.
 * Sets to fp4 tolerances if any fp4x2_type is found.
 * Else sets to fp16 tolerances if `quantize` input is fp16 or any fp16 instruction is found in the
 * model.
 */
verify::tolerance get_tolerances(const program& p,
                                 const verify_options& vo,
                                 std::optional<double> rms_tol,
                                 std::optional<double> atol,
                                 std::optional<double> rtol)
{
    bool has_16bit = any_of(p.get_modules(), [](auto&& m) {
        return any_of(*m, [](auto&& ins) {
            return (ins.get_shape().type() == shape::half_type or
                    ins.get_shape().type() == shape::bf16_type);
        });
    });
    bool has_fp4   = any_of(p.get_modules(), [](auto&& m) {
        return any_of(*m, [](auto&& ins) { return (ins.get_shape().type() == shape::fp4x2_type); });
    });
    migraphx::verify::tolerance result{};
    if(has_fp4)
    {
        result.rms_tol = 8e-1;
        result.atol    = 4e-1;
        result.rtol    = 4e-1;
    }
    else if(has_16bit or vo.quantize == precision::fp16 or vo.quantize == precision::bf16)
    {
        result.rms_tol = 8e-2;
        result.atol    = 4e-2;
        result.rtol    = 4e-2;
    }
    if(rms_tol)
    {
        result.rms_tol = *rms_tol;
    }
    if(atol)
    {
        result.atol = *atol;
    }
    if(rtol)
    {
        result.rtol = *rtol;
    }
    return result;
}

namespace {

verify::program_options make_program_options(const compile_options& options,
                                             const verify_options& vo,
                                             verify::tolerance tols)
{
    verify::program_options result{
        .compile = options, .tols = tols, .ref_use_double = vo.ref_use_double};
    result.compiled_model = vo.compiled_model;
    if(vo.quantize == precision::fp16)
        result.quantize = verify::program_precision::fp16;
    else if(vo.quantize == precision::bf16)
        result.quantize = verify::program_precision::bf16;
    return result;
}

} // namespace

bool verify_program(const std::string& name,
                    const program& p,
                    const target& t,
                    const compile_options& options,
                    const verify_options& vo,
                    const parameter_map& inputs,
                    verify::tolerance tols)
{
    auto opts = make_program_options(options, vo, tols);
    opts.name = name;
    return verify::verify_program(p, t, verify::program_mode::outputs, inputs, opts).passed();
}

void verify_instructions(const program& prog,
                         const target& t,
                         const compile_options& options,
                         const verify_options& vo,
                         verify::tolerance tols)
{
    verify::verify_program(
        prog, t, verify::program_mode::instructions, {}, make_program_options(options, vo, tols));
}

void verify_reduced_program(const program& p,
                            const target& t,
                            const compile_options& options,
                            const verify_options& vo,
                            const parameter_map& inputs,
                            verify::tolerance tols)
{
    verify::verify_program(
        p, t, verify::program_mode::reduce, inputs, make_program_options(options, vo, tols));
}

void verify_bisected_program(const program& p,
                             const target& t,
                             const compile_options& options,
                             const verify_options& vo,
                             const parameter_map& inputs,
                             verify::tolerance tols)
{
    verify::verify_program(
        p, t, verify::program_mode::bisect, inputs, make_program_options(options, vo, tols));
}

void verify_layerwise_program(const program& p,
                              const target& t,
                              const compile_options& options,
                              const verify_options& vo,
                              const parameter_map& inputs,
                              verify::tolerance tols)
{
    verify::verify_program(
        p, t, verify::program_mode::layerwise, inputs, make_program_options(options, vo, tols));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
