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

#include <onnx_test.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/external_data.hpp>
#include <migraphx/mapped_file.hpp>
#include <algorithm>

TEST_CASE(external_data_scalar_test)
{
    auto prog = read_onnx("external_data_scalar_test.onnx");
    auto* mm  = prog.get_main_module();
    auto lit  = std::find_if(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "@literal";
    });
    EXPECT(lit != mm->end());
    EXPECT(lit->get_shape() == migraphx::shape{migraphx::shape::float_type});
    EXPECT(not lit->get_literal().is_external());
    EXPECT(lit->get_literal() ==
           migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {2.5f}});
}

TEST_CASE(external_data_scalar_reuse_test)
{
    migraphx::onnx_options options;
    options.reuse_external_data = true;
    auto prog                   = read_onnx("external_data_scalar_test.onnx", options);
    auto* mm                    = prog.get_main_module();
    auto lit = std::find_if(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "@literal";
    });
    EXPECT(lit != mm->end());
    EXPECT(lit->get_shape() == migraphx::shape{migraphx::shape::float_type});
    const auto& l = lit->get_literal();
    EXPECT(l.is_external());
    EXPECT(l.get_external_data()->size == sizeof(float));
    EXPECT(migraphx::fs::path{l.get_external_data()->file_path}.filename() ==
           "external_data_scalar_test.weight");
    EXPECT(l == migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {2.5f}});
}
