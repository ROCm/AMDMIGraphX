/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_DRIVER_VERIFY_HPP
#define MIGRAPHX_GUARD_RTGLIB_DRIVER_VERIFY_HPP

#include "verify_options.hpp"
#include <migraphx/argument.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/program.hpp>
#include <migraphx/verify.hpp>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

verify::tolerance get_tolerances(const program& p,
                                 const verify_options& vo,
                                 std::optional<double> rms_tol,
                                 std::optional<double> atol,
                                 std::optional<double> rtol);

// Captures ref outputs by debug symbol and compares each target op at its terminal symbol.
struct verify_callback
{
    using trace_function = std::function<void(instruction_ref, const argument&)>;

    struct layer_result
    {
        std::string symbol = {};
        std::string op     = {};
        std::size_t order  = 0;
        double rms_error   = 0;
        double introduced  = 0;
        bool passed        = false;
    };

    struct ref_output
    {
        argument output   = {};
        std::size_t order = 0;
    };
    struct target_output
    {
        argument output                 = {};
        std::string op                  = {};
        std::size_t order               = 0;
        std::vector<std::string> inputs = {};
    };

    verify::tolerance tols = {};

    std::size_t ref_count                                         = 0;
    std::unordered_map<std::string, ref_output> ref_outputs       = {};
    std::unordered_map<std::string, target_output> target_outputs = {};
    std::unordered_map<std::string, layer_result> results         = {};

    // Latest-reference symbol of ins and its order; empty if none was traced.
    std::pair<std::string, std::size_t> terminal(instruction_ref ins) const;

    trace_function capture();
    trace_function compare();
    void evaluate();

    // The op introducing the most error over its inputs (the divergence source).
    optional<layer_result> source_failure() const;
};

bool verify_program(const std::string& name,
                    const program& p,
                    const target& t,
                    compile_options options     = compile_options{},
                    const verify_options& vo    = verify_options{},
                    const parameter_map& inputs = {},
                    verify::tolerance tols      = verify::tolerance{});
void verify_instructions(const program& prog,
                         const target& t,
                         compile_options options  = compile_options{},
                         const verify_options& vo = verify_options{},
                         verify::tolerance tols   = verify::tolerance{});
void verify_reduced_program(const program& p,
                            const target& t,
                            compile_options options     = compile_options{},
                            const verify_options& vo    = verify_options{},
                            const parameter_map& inputs = {},
                            verify::tolerance tols      = verify::tolerance{});
void verify_bisected_program(const program& p,
                             const target& t,
                             compile_options options     = compile_options{},
                             const verify_options& vo    = verify_options{},
                             const parameter_map& inputs = {},
                             verify::tolerance tols      = verify::tolerance{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
