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
#ifndef MIGRAPHX_GUARD_PROGRAM_VERIFY_HPP
#define MIGRAPHX_GUARD_PROGRAM_VERIFY_HPP

#include <migraphx/compile_options.hpp>
#include <migraphx/config.hpp>
#include <migraphx/program.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/target.hpp>
#include <migraphx/verify.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace verify {

enum class program_mode : std::uint8_t
{
    outputs,
    instructions,
    reduce,
    bisect,
    layerwise
};

enum class program_precision
{
    fp32,
    fp16,
    bf16
};

struct program_options
{
    compile_options compile    = {};
    tolerance tols             = {};
    program_precision quantize = program_precision::fp32;
    bool ref_use_double        = false;
    std::string compiled_model = {};
    std::string name           = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.compile.offload_copy, "offload_copy"),
                    f(self.compile.fast_math, "fast_math"),
                    f(self.compile.exhaustive_tune, "exhaustive_tune"),
                    f(self.compile.compile_mode, "compile_mode"),
                    f(self.compile.backend_options, "advance_backend_options"),
                    f(self.tols.rms_tol, "rms_tol"),
                    f(self.tols.atol, "atol"),
                    f(self.tols.rtol, "rtol"),
                    f(self.quantize, "precision"),
                    f(self.ref_use_double, "ref_use_double"),
                    f(self.compiled_model, "compiled_model"),
                    f(self.name, "name"));
    }
};

struct layer_result
{
    std::string name    = {};
    std::string op      = {};
    std::string message = {};
    std::size_t index   = 0;
    double rms_error    = 0;
    bool passed         = false;
    bool exception      = false;
};

struct program_result
{
    program_mode mode                       = program_mode::outputs;
    bool success                            = true;
    std::vector<layer_result> results       = {};
    std::optional<std::size_t> failure_step = {};

    MIGRAPHX_EXPORT bool passed() const;
    MIGRAPHX_EXPORT std::vector<layer_result> failures() const;
};

MIGRAPHX_EXPORT program_result verify_program(const program& p,
                                              const target& t,
                                              program_mode mode           = program_mode::outputs,
                                              const parameter_map& inputs = {},
                                              const program_options& options = {});

} // namespace verify
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
