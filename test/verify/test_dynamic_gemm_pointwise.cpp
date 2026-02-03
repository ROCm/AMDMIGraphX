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
#include <migraphx/instruction.hpp>
#include <migraphx/env.hpp>
#include <test.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR)

template <std::size_t B, std::size_t K>
struct test_dynamic_gemm_pointwise : verify_program<test_dynamic_gemm_pointwise<B, K>>
{
    migraphx::program create_program() const
    {
        // Skip test when MLIR is disabled
        if(migraphx::enabled(MIGRAPHX_DISABLE_MLIR{}))
            test::skip("MIGRAPHX_DISABLE_MLIR is set");

        migraphx::shape s_x{migraphx::shape::float_type, {{2, 4}, {3, 3}, {4, 24}}};
        migraphx::shape s_w{migraphx::shape::float_type, {{2, 4}, {4, 24}, {5, 5}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 3, 5}}));
        auto gemm = mm->add_instruction(migraphx::make_op("dot"), x, w);
        auto b_bc = mm->add_instruction(migraphx::make_op("multibroadcast"), b, gemm);
        auto add  = mm->add_instruction(migraphx::make_op("add"), gemm, b_bc);
        auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
        mm->add_return({relu});
        return p;
    };

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"x", migraphx::shape{migraphx::shape::float_type, {B, 3, K}}},
                {"w", migraphx::shape{migraphx::shape::float_type, {B, K, 5}}}};
    }
};

template struct test_dynamic_gemm_pointwise<4, 4>;
template struct test_dynamic_gemm_pointwise<3, 24>;
template struct test_dynamic_gemm_pointwise<2, 16>;
