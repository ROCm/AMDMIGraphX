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
#include <migraphx/ranges.hpp>

struct test_unary_math : verify_programs<test_unary_math>
{
    using programs = std::vector<std::pair<std::string, std::function<migraphx::program()>>>;
    auto generate_program(const std::string& name, migraphx::shape::type_t t) const
    {
        return [=] {
            migraphx::program p;
            auto* mm = p.get_main_module();
            migraphx::shape s{t, {10}};
            auto x = mm->add_parameter("x", s);
            mm->add_instruction(migraphx::make_op(name), x);
            return p;
        };
    }

    std::string section() const { return "test_math"; };

    programs get_programs() const
    {
        programs ps;
        const std::vector<std::string> names = {
            // clang-format off
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "ceil",
            "cos",
            "cosh",
            "erf",
            "exp",
            "floor",
            "isnan",
            "log",
            "round",
            "rsqrt",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            // clang-format on
        };
        for(auto&& t : migraphx::shape::types())
        {
            if(migraphx::contains({migraphx::shape::bool_type, migraphx::shape::tuple_type}, t))
                continue;
            for(const auto& name : names)
            {
                std::string test_name = "test_math_" + name + "_" + migraphx::shape::cpp_type(t);
                ps.push_back(std::make_pair(test_name, generate_program(name, t)));
            }
        }
        return ps;
    }
};
