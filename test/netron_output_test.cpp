/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/netron_output.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx.hpp>
#include <sstream>
#include <test.hpp>

TEST_CASE(netron_output_basic)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    std::string output = os.str();
    EXPECT(not output.empty());
    EXPECT(output.size() > 10);
}

TEST_CASE(netron_output_with_literal)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto lit = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {2, 3}},
                                                  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, lit);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);

    std::string output = os.str();
    EXPECT(not output.empty());
    EXPECT(output.size() > 10);
}

TEST_CASE(netron_output_roundtrip)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto sum = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_return({sum});

    std::ostringstream os;
    migraphx::write_netron_output(p, os);
    std::string output = os.str();

    // The output should be parseable as a valid ONNX model
    migraphx::onnx_options options;
    options.skip_unknown_operators = true;
    auto p2 = migraphx::parse_onnx_buffer(output.data(), output.size(), options);
    EXPECT(p2.get_main_module()->size() > 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
