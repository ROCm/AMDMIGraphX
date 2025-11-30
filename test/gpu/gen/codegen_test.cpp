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
#include <test.hpp>
#include <migraphx/gpu/gen/codegen.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>

TEST_CASE(test_generate_pointwise_kernel)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type});
    auto add_op = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_return({add_op});

    auto code = migraphx::gpu::gen::generate_pointwise_kernel(*mm, "test_kernel");
    EXPECT(not code.empty());
    EXPECT(migraphx::contains(code, "test_kernel"));
}

TEST_CASE(test_generate_unary_kernel)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type});
    auto neg_op = mm->add_instruction(migraphx::make_op("neg"), x);
    mm->add_return({neg_op});

    auto code = migraphx::gpu::gen::generate_pointwise_kernel(*mm, "neg_kernel");
    EXPECT(not code.empty());
    EXPECT(migraphx::contains(code, "neg_kernel"));
}

TEST_CASE(test_generate_complex_pointwise_kernel)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type});
    auto add_res = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto mul_res = mm->add_instruction(migraphx::make_op("mul"), add_res, x);
    mm->add_return({mul_res});

    auto code = migraphx::gpu::gen::generate_pointwise_kernel(*mm, "fused_kernel");
    EXPECT(not code.empty());
    EXPECT(migraphx::contains(code, "fused_kernel"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
