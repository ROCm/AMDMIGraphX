/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>
// template <migraphx::shape::type_t DType>
// struct test_resize_dyn : verify_program<test_resize_dyn<DType>>

//  TODO:  is this template correct for this test, or the one above?  Unknown error that the test 
//    isn't found if I use the above template.
struct test_resize_dyn : verify_program<test_resize_dyn>
{
    // TODO:  This test causes an assertion failure in propagate_constant.cpp.  Need to find
    //    out why.
    migraphx::program create_program() const
    {
        
   // matcher/optimized code should produce the same result as Resize op.
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 3, 3}};
    auto a0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape size_input{migraphx::shape::int32_type, {4}};
    std::vector<int> size_values = {1, 1, 5, 8};
    auto a1                      = mm->add_literal(migraphx::literal{size_input, size_values});

    // a0 = input data
    // a1 = sizes of output
    // non-matching sizes/scales attributes are ignored for 2 input arguments
    auto resize_ins = mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {1}},
                                           {"scales", {1}},
                                           {"nearest_mode", "floor"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        a0,
                        a1);
    // mm->add_return({resize_ins});
    return p;

    };
};
// template struct test_resize_dyn<migraphx::shape::float_type>;
