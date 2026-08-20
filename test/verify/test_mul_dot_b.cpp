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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/fp8_types.hpp>

template <migraphx::shape::type_t DType>
struct test_mul_dot_b : verify_program<test_mul_dot_b<DType>>

{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape as{DType, {2, 256, 32}};
        migraphx::shape bs{DType, {2, 32, 128}};
        auto b = mm->add_parameter("input", bs);
        auto lit  = mm->add_literal(migraphx::generate_literal({DType, {1, 32, 1}}));
        auto litb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bs.lens()}}), lit);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), b, litb);
        auto a   = mm->add_literal(migraphx::generate_literal(as));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, mul);
        mm->add_return({dot});
        return p;
    }
    std::string section() const { return "gemm"; }

    // The dot itself already accumulates in double on the ref, but the elementwise multiply
    // ahead of it is rounded in the storage type there while the gpu fuses it.
    bool get_ref_use_double() const { return true; }

    migraphx::optional<migraphx::verify::tolerance> get_tolerance() const
    {
        // A handful of the 65536 dot outputs cancel to near zero, where only atol reaches them.
        // Measured: atol alone needs 1.6x.
        if(migraphx::contains(migraphx::fp8_types{}.get(), DType))
        {
            auto tols = migraphx::default_tolerance_for(DType);
            tols.atol *= 4;
            return tols;
        }
        return migraphx::nullopt;
    }
};

template struct test_mul_dot_b<migraphx::shape::float_type>;
template struct test_mul_dot_b<migraphx::shape::half_type>;
template struct test_mul_dot_b<migraphx::shape::bf16_type>;
template struct test_mul_dot_b<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_mul_dot_b<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_mul_dot_b<migraphx::shape::fp8e4m3fn_type>;
template struct test_mul_dot_b<migraphx::shape::fp8e5m2_type>;
