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
#include <migraphx/cpu/target.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/program_verify.hpp>
#include <test.hpp>

// The cpu outputs are views into memory owned by the compiled program, so they
// must stay valid until they are compared with the ref outputs.
TEST_CASE(verify_program_outputs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto a   = mm->add_parameter("a", migraphx::shape{migraphx::shape::float_type, {4, 5}});
    auto b   = mm->add_parameter("b", migraphx::shape{migraphx::shape::float_type, {5, 3}});
    mm->add_return({mm->add_instruction(migraphx::make_op("dot"), a, b)});

    auto result = migraphx::verify::verify_program(
        p, migraphx::cpu::target{}, migraphx::verify::program_mode::outputs);
    EXPECT(result.passed());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
