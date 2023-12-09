/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>


TEST_CASE(resize_test_1)
{
    // batch size 1, 1 color channel, resize 3x3 to 5x8
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    // to do: non-literal
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1  = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    mm->add_instruction(migraphx::make_op("resize", {{"sizes", {1}}, {"scales", {}}, {"nearest_mode", "floor"}
      , {"coordinate_transformation_mode", "half_pixel"}}), a0, a1);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> res_data(1*1*5*8);
    std::vector<float> golden = {0.5f, 1.5f, 2.5f, 6.5f, 7.5f, 8.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
    for(auto aa : res_data) std::cout << aa << ", "; std::cout << " result \n";
    EXPECT(migraphx::verify::verify_rms_range(res_data, golden));
}
