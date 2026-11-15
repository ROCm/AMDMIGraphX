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

TEST_CASE(slice_select_less_symbolic_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    // The end is to the end of the axis while N < 50, and 50 otherwise
    const auto n = var("N", {1, 300});
    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] =
        migraphx::shape{migraphx::shape::float_type, sym_dims({n, lit(4)})}.dyn_dims();
    options.use_symbolic_shapes = true;
    auto p                      = read_onnx("slice_select_less_symbolic_test.onnx", options);
    auto* mm                    = p.get_main_module();
    auto slice                  = std::find_if(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "dyn_slice"; });
    EXPECT(slice != mm->end());
    if(slice == mm->end())
        return;
    auto bounds = migraphx::from_value<std::vector<migraphx::sym::expr>>(
        slice->get_operator().to_value().at("ends"));
    EXPECT(bounds.size() == 1);
    if(bounds.size() != 1)
        return;
    EXPECT(bounds[0] == migraphx::sym::min(n, lit(50)));
}
