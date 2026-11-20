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

#include <migraphx/iterator_for.hpp>
#include <onnx_test.hpp>

// The first Slice gives a symbolic output, so the second Slice must also use dyn_slice.
TEST_CASE(slice_var_ends_twice_test)
{
    auto prog = read_onnx("slice_var_ends_twice_test.onnx");
    auto* mm  = prog.get_main_module();

    std::vector<migraphx::instruction_ref> slices;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        EXPECT(ins->name() != "slice");
        if(ins->name() == "dyn_slice")
            slices.push_back(ins);
    }
    EXPECT(slices.size() == 2);
    EXPECT(slices.back()->inputs().front() == slices.front());
    for(auto ins : slices)
    {
        EXPECT(ins->get_shape().symbolic());
        EXPECT(ins->get_shape().max_lens() == std::vector<std::size_t>{10, 4});
    }
}
