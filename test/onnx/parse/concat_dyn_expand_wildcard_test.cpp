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
#include <algorithm>

TEST_CASE(concat_dyn_expand_wildcard_test)
{
    // Regression for #4924. ONNX Expand(qu, dims) with a runtime `dims` lowers
    // to broadcast_with_dims, whose output carries fully-unconstrained dynamic
    // dims. Concatenating that with the dynamic `item` previously threw
    // "CONCAT: all input dimensions should match in axis 0" while parsing
    // (add_instruction runs concat shape inference). #4924 treats the
    // unconstrained dim as a wildcard, so the model now parses.
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = read_onnx("concat_dyn_expand_wildcard_test.onnx", options);

    auto* mm = prog.get_main_module();
    EXPECT(std::any_of(mm->begin(), mm->end(), [](const auto& ins) {
        return ins.name() == "broadcast_with_dims";
    }));
    EXPECT(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "concat"; }));

    auto out_shapes = prog.get_output_shapes();
    EXPECT(out_shapes.size() == 1);
    EXPECT(out_shapes.front().dynamic());
    EXPECT(out_shapes.front().ndim() == 2);
}
