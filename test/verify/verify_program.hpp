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
#ifndef MIGRAPHX_GUARD_AUTO_REGISTER_VERIFY_PROGRAM_HPP
#define MIGRAPHX_GUARD_AUTO_REGISTER_VERIFY_PROGRAM_HPP

#include <migraphx/auto_register.hpp>
#include <migraphx/program.hpp>
#include <functional>

struct program_info
{
    std::string name;
    std::string section;
    std::function<migraphx::program()> get_program;
};

void register_program_info(const program_info& pi);
const std::vector<program_info>& get_programs();

struct register_verify_program_action
{
    template <class T>
    static void apply()
    {
        T x;
        program_info pi;
        pi.name        = migraphx::get_type_name<T>();
        pi.section     = x.section();
        pi.get_program = [x] { return x.create_program(); };
        register_program_info(pi);
    }
};

template <class T>
using auto_register_verify_program = migraphx::auto_register<register_verify_program_action, T>;

template <class T>
struct verify_program : auto_register_verify_program<T>
{
    std::string section() const { return "general"; };
};

#endif
