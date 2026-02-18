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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>

// Template parameters:
// - T: argmin or argmax op type
// - DType: data type (float_type, half_type, etc.)
// - Axis: reduction axis
// - FusionType: 0=add, 1=mul, 2=relu, 3=neg, 4=abs, 5=none (just arg op)
template <class T, migraphx::shape::type_t DType, int Axis, int FusionType>
struct test_arg_fusion : verify_program<test_arg_fusion<T, DType, Axis, FusionType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 3, 4}};
        auto x = mm->add_parameter("x", s);

        migraphx::instruction_ref input;
        switch(FusionType)
        {
        case 0: // add fusion
        {
            auto y = mm->add_parameter("y", s);
            input  = mm->add_instruction(migraphx::make_op("add"), x, y);
            break;
        }
        case 1: // mul fusion
        {
            auto y = mm->add_parameter("y", s);
            input  = mm->add_instruction(migraphx::make_op("mul"), x, y);
            break;
        }
        case 2: // relu fusion
            input = mm->add_instruction(migraphx::make_op("relu"), x);
            break;
        case 3: // neg fusion
            input = mm->add_instruction(migraphx::make_op("neg"), x);
            break;
        case 4: // abs fusion
            input = mm->add_instruction(migraphx::make_op("abs"), x);
            break;
        default: // no fusion, just arg op
            input = x;
            break;
        }

        mm->add_instruction(T{Axis, false}, input);
        return p;
    }

    std::string section() const { return "reduce"; }
};

// argmin with add fusion, different axes
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 0, 0>;
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 1, 0>;
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 2, 0>;
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, -1, 0>;

// argmax with add fusion, different axes
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 0, 0>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 1, 0>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 2, 0>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, -1, 0>;

// argmin with mul fusion
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 1, 1>;
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 2, 1>;

// argmax with mul fusion
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 1, 1>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 2, 1>;

// argmin with relu fusion
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 0, 2>;
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 1, 2>;

// argmax with neg fusion
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 1, 3>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 2, 3>;

// argmax with abs fusion
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 1, 4>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 2, 4>;

// half precision tests
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::half_type, 1, 0>;
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::half_type, 2, 0>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::half_type, 1, 0>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::half_type, 2, 0>;

// no fusion (just arg op) tests
template struct test_arg_fusion<migraphx::op::argmin, migraphx::shape::float_type, 1, 5>;
template struct test_arg_fusion<migraphx::op::argmax, migraphx::shape::float_type, 1, 5>;

// Additional tests with variable shapes (don't fit template pattern)

// Test argmin on axis 0 with different shape {4, 3, 2}
struct test_arg_fusion_argmin_shape_4_3_2 : verify_program<test_arg_fusion_argmin_shape_4_3_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {4, 3, 2}};
        auto x    = mm->add_parameter("x", s);
        auto relu = mm->add_instruction(migraphx::make_op("relu"), x);
        mm->add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), relu);
        return p;
    }

    std::string section() const { return "reduce"; }
};

// Test argmax on last axis with larger reduction {8, 5, 256}
struct test_arg_fusion_argmax_shape_8_5_256 : verify_program<test_arg_fusion_argmax_shape_8_5_256>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {8, 5, 256}};
        auto x   = mm->add_parameter("x", s);
        auto neg = mm->add_instruction(migraphx::make_op("neg"), x);
        mm->add_instruction(migraphx::make_op("argmax", {{"axis", -1}}), neg);
        return p;
    }

    std::string section() const { return "reduce"; }
};

// Test argmin with broadcast input fusion
struct test_arg_fusion_argmin_broadcast : verify_program<test_arg_fusion_argmin_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
        migraphx::shape bs{migraphx::shape::float_type, {1, 3, 4}};
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", bs);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), y);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, yb);
        mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), add);
        return p;
    }

    std::string section() const { return "reduce"; }
};

// Test argmax with broadcast input fusion
struct test_arg_fusion_argmax_broadcast : verify_program<test_arg_fusion_argmax_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
        migraphx::shape bs{migraphx::shape::float_type, {2, 1, 4}};
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", bs);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), y);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), x, yb);
        mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), mul);
        return p;
    }

    std::string section() const { return "reduce"; }
};
