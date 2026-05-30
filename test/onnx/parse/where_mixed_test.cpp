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

TEST_CASE(where_mixed_test)
{
    // Mixed static + dynamic where inputs are now broadcast to a common
    // shape via add_common_op (previously unsupported -- threw). The static
    // input y={3,2,2} pins the broadcasted dimension, so the result has
    // fixed dims {3,2,2}.
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = read_onnx("where_mixed_test.onnx", options);

    auto out_shapes = prog.get_output_shapes();
    EXPECT(out_shapes.size() == 1);
    migraphx::shape expected{migraphx::shape::float_type, {{3, 3}, {2, 2}, {2, 2}}};
    EXPECT(out_shapes.front() == expected);
}
