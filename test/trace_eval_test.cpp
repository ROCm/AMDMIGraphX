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

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <migraphx/compile_options.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include "test.hpp"

// MIGRAPHX_TRACE_EVAL is process-cached on first read; set in main() before
// any program::eval call. trace_level == 2 enables per-instruction trace
// output and an internal exec_env.trace lambda that calls copy_to_host().

struct test_target
{
    struct context
    {
        void finish() const {}
    };
    migraphx::context ctx = context{};
    std::string name() const { return "test"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {};
    }
    migraphx::context get_context() const { return ctx; }
};

struct migraphx_throw_target : test_target
{
    migraphx::argument copy_from(const migraphx::argument&) const { MIGRAPHX_THROW("x"); }
};

struct std_throw_target : test_target
{
    migraphx::argument copy_from(const migraphx::argument&) const { throw std::runtime_error("x"); }
};

template <class Target>
static migraphx::program make_add_program(Target t)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto a   = mm->add_literal(1);
    auto b   = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), a, b);
    p.compile(std::move(t));
    return p;
}

struct cout_redirect
{
    explicit cout_redirect(std::stringstream& s) : old(std::cout.rdbuf(s.rdbuf())) {}
    ~cout_redirect() { std::cout.rdbuf(old); }
    std::streambuf* old;
};

TEST_CASE(trace_eval_level_2)
{
    auto p = make_add_program(test_target{});
    std::stringstream out;
    {
        cout_redirect cr{out};
        p.eval({});
    }
    EXPECT(out.str().find("Run instruction") != std::string::npos);
    EXPECT(out.str().find("Output has") != std::string::npos);
}

TEST_CASE(trace_eval_migraphx_exception)
{
    auto p = make_add_program(migraphx_throw_target{});
    std::stringstream out;
    cout_redirect cr{out};
    EXPECT(p.eval({}).back() == migraphx::literal{3});
}

TEST_CASE(trace_eval_std_exception)
{
    auto p = make_add_program(std_throw_target{});
    std::stringstream out;
    cout_redirect cr{out};
    EXPECT(test::throws<migraphx::exception>([&] { p.eval({}); }, "Failed to copy result to host"));
}

int main(int argc, const char* argv[])
{
#ifdef _WIN32
    _putenv_s("MIGRAPHX_TRACE_EVAL", "2");
#else
    setenv("MIGRAPHX_TRACE_EVAL", "2", 1);
#endif
    test::run(argc, argv);
}
