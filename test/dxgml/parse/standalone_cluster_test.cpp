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
#include <dxgml_test.hpp>
#include <migraphx/instruction.hpp>

// StandaloneCluster.CompilationInput.mlir:
//   entry: %arg0: half[1,4,2160,3840]
//   Pre-conv: relu, add(r0,r0), multiply(r0,a0)
//   Constants: _conv1.weight half[32,4,3,3], _conv1.bias half[32]
//   Conv: stride=2, dilation=1, pad=1, groups=1
//   Post-conv: relu, add(r1,r1), multiply(a1,a1)
//   return m1
TEST_CASE(standalone_cluster_parse_test)
{
    auto prog = read_dxgml("StandaloneCluster.CompilationInput.mlir");
    auto* mm  = prog.get_main_module();

    // arg0 + relu + add + mul + weight + bias + conv + relu + add + mul + return = 11
    std::size_t n = std::distance(mm->begin(), mm->end());
    EXPECT(n == 11);

    auto at = [&](std::size_t i) { return std::next(mm->begin(), static_cast<std::ptrdiff_t>(i)); };

    // [0] @param arg0  half[1,4,2160,3840]
    EXPECT(at(0)->name() == "@param");
    EXPECT(at(0)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {1, 4, 2160, 3840}});

    // [1] relu (pre-conv)
    EXPECT(at(1)->name() == "relu");
    EXPECT(at(1)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {1, 4, 2160, 3840}});

    // [2] add
    EXPECT(at(2)->name() == "add");

    // [3] mul (pre-conv)
    EXPECT(at(3)->name() == "mul");

    // [4] @param _conv1.weight  half[32,4,3,3]
    EXPECT(at(4)->name() == "@param");
    EXPECT(at(4)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {32, 4, 3, 3}});

    // [5] @param _conv1.bias  half[32]
    EXPECT(at(5)->name() == "@param");
    EXPECT(at(5)->get_shape() == migraphx::shape{migraphx::shape::half_type, {32}});

    // [6] convolution -> half[1,32,1080,1920]
    EXPECT(at(6)->name() == "convolution");
    EXPECT(at(6)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {1, 32, 1080, 1920}});

    // [7] relu (post-conv)
    EXPECT(at(7)->name() == "relu");

    // [8] add
    EXPECT(at(8)->name() == "add");

    // [9] mul (post-conv)
    EXPECT(at(9)->name() == "mul");
    EXPECT(at(9)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {1, 32, 1080, 1920}});

    // [10] @return
    EXPECT(at(10)->name() == "@return");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
