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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// This covers split-k perfConfigs that reject conv+pointwise+layout fusion. The add is still
// absorbed into the MLIR kernel, leaving a conv+add kernel and a layout-copy kernel.
struct test_conv_add_reshape_lazy_transpose
    : verify_program<test_conv_add_reshape_lazy_transpose>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto input  = mm->add_parameter("x", {migraphx::shape::half_type, {1, 256, 16, 16}});
        auto weight = mm->add_literal(
            migraphx::generate_literal({migraphx::shape::half_type, {1, 256, 3, 2}}, 1));
        auto y = mm->add_parameter("y", {migraphx::shape::half_type, {1, 1, 8, 8}});

        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 0}}, {"stride", {2, 2}}}),
            input,
            weight);
        auto add = mm->add_instruction(migraphx::make_op("add"), conv, y);

        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", {1, 1, 4, 2, 8}}}), add);
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2, 4}}}),
                            reshape);
        return p;
    }

    // Turn on Exhaustive-tune to enable split-k perf-configs from MLIR
    migraphx::compile_options get_compile_options() const
    {
        return migraphx::compile_options{.exhaustive_tune = true};
    }

    std::string section() const { return "conv"; }
};
