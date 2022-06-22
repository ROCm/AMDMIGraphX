/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>

template <class T, int Axis, int NonStdShape>
struct test_arg_ops : verify_program<test_arg_ops<T, Axis, NonStdShape>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 1, 4, 1025}};
        auto param = mm->add_parameter("data", s);
        switch(NonStdShape)
        {
        case 0:
            param = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), param);
            break;
        case 1:
            param = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 1025}}}), param);
            break;
        case 2:
            param = mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}), param);
            break;
        default: break;
        }
        mm->add_instruction(T{Axis}, param);
        return p;
    }
};
// transpose argmax tests
template struct test_arg_ops<migraphx::op::argmax, 0, 0>;
template struct test_arg_ops<migraphx::op::argmax, 1, 0>;
template struct test_arg_ops<migraphx::op::argmax, 2, 0>;
template struct test_arg_ops<migraphx::op::argmax, 3, 0>;
template struct test_arg_ops<migraphx::op::argmax, -1, 0>;
template struct test_arg_ops<migraphx::op::argmax, -2, 0>;
// transpose argmin tests
template struct test_arg_ops<migraphx::op::argmin, 0, 0>;
template struct test_arg_ops<migraphx::op::argmin, 1, 0>;
template struct test_arg_ops<migraphx::op::argmin, 2, 0>;
template struct test_arg_ops<migraphx::op::argmin, 3, 0>;
template struct test_arg_ops<migraphx::op::argmin, -3, 0>;
template struct test_arg_ops<migraphx::op::argmin, -4, 0>;
// broadcast argmax tests
template struct test_arg_ops<migraphx::op::argmax, 0, 1>;
template struct test_arg_ops<migraphx::op::argmax, 1, 1>;
template struct test_arg_ops<migraphx::op::argmax, 2, 1>;
template struct test_arg_ops<migraphx::op::argmax, 3, 1>;
template struct test_arg_ops<migraphx::op::argmax, -1, 1>;
template struct test_arg_ops<migraphx::op::argmax, -2, 1>;
// broadcast argmin tests
template struct test_arg_ops<migraphx::op::argmin, 0, 1>;
template struct test_arg_ops<migraphx::op::argmin, 1, 1>;
template struct test_arg_ops<migraphx::op::argmin, 2, 1>;
template struct test_arg_ops<migraphx::op::argmin, 3, 1>;
template struct test_arg_ops<migraphx::op::argmin, -3, 1>;
template struct test_arg_ops<migraphx::op::argmin, -4, 1>;
// slice argmax tests
template struct test_arg_ops<migraphx::op::argmax, 0, 2>;
template struct test_arg_ops<migraphx::op::argmax, 1, 2>;
template struct test_arg_ops<migraphx::op::argmax, 2, 2>;
template struct test_arg_ops<migraphx::op::argmax, 3, 2>;
template struct test_arg_ops<migraphx::op::argmax, -1, 2>;
template struct test_arg_ops<migraphx::op::argmax, -2, 2>;
// slice argmin tests
template struct test_arg_ops<migraphx::op::argmin, 0, 2>;
template struct test_arg_ops<migraphx::op::argmin, 1, 2>;
template struct test_arg_ops<migraphx::op::argmin, 2, 2>;
template struct test_arg_ops<migraphx::op::argmin, 3, 2>;
template struct test_arg_ops<migraphx::op::argmin, -3, 2>;
template struct test_arg_ops<migraphx::op::argmin, -4, 2>;
// default case, standard shape argmax tests
template struct test_arg_ops<migraphx::op::argmax, 0, 3>;
template struct test_arg_ops<migraphx::op::argmax, 1, 3>;
template struct test_arg_ops<migraphx::op::argmax, 2, 3>;
template struct test_arg_ops<migraphx::op::argmax, 3, 3>;
template struct test_arg_ops<migraphx::op::argmax, -1, 3>;
template struct test_arg_ops<migraphx::op::argmax, -2, 3>;
// default case, standard shape argmin tests
template struct test_arg_ops<migraphx::op::argmin, 0, 3>;
template struct test_arg_ops<migraphx::op::argmin, 1, 3>;
template struct test_arg_ops<migraphx::op::argmin, 2, 3>;
template struct test_arg_ops<migraphx::op::argmin, 3, 3>;
template struct test_arg_ops<migraphx::op::argmin, -3, 3>;
template struct test_arg_ops<migraphx::op::argmin, -4, 3>;
