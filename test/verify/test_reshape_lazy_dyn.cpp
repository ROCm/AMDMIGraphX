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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

template <std::size_t B>
struct test_reshape_lazy_dyn : verify_program<test_reshape_lazy_dyn<B>>
{
    migraphx::program create_program() const
    {
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto input  = mm->add_parameter("X", s);
        auto result = mm->add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", std::vector<int64_t>{0, 8, 3, 1}}}), input);
        mm->add_return({result});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"X", migraphx::shape{migraphx::shape::float_type, {B, 24, 1, 1}}}};
    }
};

template struct test_reshape_lazy_dyn<4>;
template struct test_reshape_lazy_dyn<2>;
template struct test_reshape_lazy_dyn<3>;

template <std::size_t B>
struct test_reshape_lazy_all_fixed_dyn : verify_program<test_reshape_lazy_all_fixed_dyn<B>>
{
    migraphx::program create_program() const
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 2}, {6, 6}}};
        migraphx::program p;
        auto* mm    = p.get_main_module();
        auto input  = mm->add_parameter("X", s);
        auto result = mm->add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", std::vector<int64_t>{1, 3, 2, 2}}}), input);
        mm->add_return({result});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"X", migraphx::shape{migraphx::shape::float_type, {2, 6}}}};
    }
};

template struct test_reshape_lazy_all_fixed_dyn<2>;
