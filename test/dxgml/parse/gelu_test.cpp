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

// Gelu.CompilationInput.mlir:
//   entry: %arg0: half[1,512,3000]
//   constants: _, __1, __2  (all half[1,512,3000])
//   arg0 / sqrt2 -> erf -> (erf + 1) -> (arg0 * add) -> (* half) -> return
//   Instruction sequence: arg0, _, __1, __2, div, erf, add, mul_input, mul_half, return
TEST_CASE(gelu_parse_test)
{
    auto prog = read_dxgml("Gelu.CompilationInput.mlir");
    auto* mm  = prog.get_main_module();

    // arg0 + 3 constants + div + erf + add + multiply + multiply + return = 10
    std::size_t n = std::distance(mm->begin(), mm->end());
    EXPECT(n == 10);

    auto at = [&](std::size_t i) { return std::next(mm->begin(), static_cast<std::ptrdiff_t>(i)); };

    // [0] @param arg0  half[1,512,3000]
    EXPECT(at(0)->name() == "@param");
    EXPECT(at(0)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {1, 512, 3000}});

    // [1], [2], [3]  constants (named parameters)
    EXPECT(at(1)->name() == "@param");
    EXPECT(at(2)->name() == "@param");
    EXPECT(at(3)->name() == "@param");

    // [4] div
    EXPECT(at(4)->name() == "div");

    // [5] erf
    EXPECT(at(5)->name() == "erf");

    // [6] add
    EXPECT(at(6)->name() == "add");

    // [7] mul (arg0 * add_result)
    EXPECT(at(7)->name() == "mul");

    // [8] mul (* half_const)
    EXPECT(at(8)->name() == "mul");
    EXPECT(at(8)->get_shape() ==
           migraphx::shape{migraphx::shape::half_type, {1, 512, 3000}});

    // [9] @return
    EXPECT(at(9)->name() == "@return");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
