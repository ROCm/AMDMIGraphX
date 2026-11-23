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
#include <onnx_test_utils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/mapped_file.hpp>
#include <algorithm>

TEST_CASE(external_data_reuse_test)
{
    migraphx::onnx_options options;
    options.reuse_external_data = true;
    auto prog                   = read_onnx("external_data_test.onnx", options);
    auto* mm                    = prog.get_main_module();
    auto last_ins               = std::prev(mm->end());
    if(last_ins->name() == "@return")
        mm->remove_instruction(last_ins);
    EXPECT(create_external_data_prog() == prog);

    EXPECT(std::count_if(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
               return ins.name() == "@literal" and ins.get_literal().is_external();
           }) == 1);
    auto weights    = std::find_if(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "@literal" and ins.get_literal().is_external();
    });
    const auto& lit = weights->get_literal();
    const auto& ext = *lit.get_external_data();
    migraphx::fs::path file_path{ext.file_path};
    EXPECT(file_path.is_absolute());
    EXPECT(file_path.filename() == "conv.weight");
    EXPECT(ext.offset == 0);
    EXPECT(ext.size == 1210 * sizeof(float));
    EXPECT(lit.get_mapped_file() != nullptr);
    EXPECT(lit.data() == lit.get_mapped_file()->data());
}
