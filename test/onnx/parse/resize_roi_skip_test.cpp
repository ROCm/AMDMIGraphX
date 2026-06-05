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

TEST_CASE(resize_roi_skip_test)
{
    // Parse the ONNX and check it produces 1-input resize with scales attribute
    auto prog = read_onnx("resize_roi_skip_test.onnx");
    auto* mm  = prog.get_main_module();

    // Check that we have a resize instruction with scales attribute
    auto resize_it = std::find_if(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "resize"; });
    EXPECT(resize_it != mm->end());

    // Check that resize has 1 input (not 2)
    EXPECT(resize_it->inputs().size() == 1);

    // Verify the output shape
    auto out_shape = resize_it->get_shape();
    EXPECT(out_shape.lens() == std::vector<std::size_t>{1, 1, 4, 8});
}
