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
#include <migraphx/ranges.hpp>
#include <onnx_test.hpp>

TEST_CASE(symbolic_conv_transpose_constant_of_shape_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {0, 100});
    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = sym_dims({batch, lit(4), lit(4), lit(4)});
    auto p   = read_onnx("symbolic_conv_transpose_constant_of_shape_test.onnx", options);
    auto& mm = *p.get_main_module();

    auto convolutions = migraphx::find_all(migraphx::iterator_for(mm), [](auto ins) {
        return ins->name() == "convolution_backwards";
    });
    EXPECT(convolutions.size() == 1);
    if(convolutions.size() == 1)
    {
        const migraphx::shape expected{migraphx::shape::float_type,
                                       sym_dims({batch, lit(3), lit(9), lit(9)})};
        EXPECT(convolutions.front()->get_shape() == expected);
    }

    auto allocations = migraphx::find_all(migraphx::iterator_for(mm),
                                          [](auto ins) { return ins->name() == "allocate"; });
    EXPECT(allocations.size() == 1);
    if(allocations.size() == 1)
    {
        const migraphx::shape expected{migraphx::shape::int64_type, sym_dims({batch})};
        EXPECT(allocations.front()->get_shape() == expected);
    }
}
