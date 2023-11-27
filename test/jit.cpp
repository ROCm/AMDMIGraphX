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
#include <migraphx/compile_src.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/cpp_generator.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

// NOLINTNEXTLINE
const std::string add_42_src = R"migraphx(
extern "C" int add(int x)
{
    return x+42;
}
)migraphx";

// NOLINTNEXTLINE
const std::string preamble = R"migraphx(
#include <cmath>
)migraphx";

template <class F>
std::function<F>
compile_function(const std::string& src, const std::string& flags, const std::string& fname)
{
    migraphx::src_compiler compiler;
    compiler.flags = flags + "-std=c++14 -shared";
#ifdef _WIN32
    compiler.output = "simple.dll";
#else
    compiler.flags += " -fPIC";
    compiler.output = "libsimple.so";
#endif
    migraphx::src_file f{"main.cpp", src};
    auto image = compiler.compile({f});
    return migraphx::dynamic_loader{image}.get_function<F>(fname);
}

template <class F>
std::function<F> compile_module(const migraphx::module& m, const std::string& flags = "")
{
    migraphx::cpp_generator g;
    g.fmap([](auto&& name) { return "std::" + name; });
    g.create_function(g.generate_module(m).set_attributes({"extern \"C\""}));

    return compile_function<F>(preamble + g.str(), flags, m.name());
}

TEST_CASE(simple_run)
{
    auto f = compile_function<int(int)>(add_42_src, "", "add");
    EXPECT(f(8) == 50);
    EXPECT(f(10) == 52);
}

TEST_CASE(generate_module)
{
    migraphx::module m("foo");
    auto x   = m.add_parameter("x", migraphx::shape::float_type);
    auto y   = m.add_parameter("y", migraphx::shape::float_type);
    auto sum = m.add_instruction(migraphx::make_op("add"), x, y);
    m.add_instruction(migraphx::make_op("sqrt"), sum);

    auto f = compile_module<float(float, float)>(m);

    EXPECT(test::within_abs(f(2, 2), 2));
    EXPECT(test::within_abs(f(10, 6), 4));
    EXPECT(test::within_abs(f(1, 2), std::sqrt(3)));
}

TEST_CASE(generate_module_with_literals)
{
    migraphx::module m("foo");
    auto x    = m.add_parameter("x", migraphx::shape::float_type);
    auto y    = m.add_parameter("y", migraphx::shape::float_type);
    auto z    = m.add_literal(1.f);
    auto sum1 = m.add_instruction(migraphx::make_op("add"), x, z);
    auto sum2 = m.add_instruction(migraphx::make_op("add"), sum1, y);
    m.add_instruction(migraphx::make_op("sqrt"), sum2);

    auto f = compile_module<float(float, float)>(m);

    EXPECT(test::within_abs(f(1, 2), 2));
    EXPECT(test::within_abs(f(9, 6), 4));
    EXPECT(test::within_abs(f(0, 2), std::sqrt(3)));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
