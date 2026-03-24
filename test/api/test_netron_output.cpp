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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <cstdio>
#include <fstream>
#include "test.hpp"

TEST_CASE(netron_output_cpp_api)
{
    auto p               = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    std::string filename = "migraphx_api_netron_output_test.onnx";
    p.write_netron_output(filename.c_str());

    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    EXPECT(ifs.good());
    auto size = ifs.tellg();
    EXPECT(size > 0);

    std::remove(filename.c_str());
}

TEST_CASE(netron_output_c_api)
{
    migraphx_program_t p;
    migraphx_onnx_options_t onnx_options;
    migraphx_onnx_options_create(&onnx_options);
    auto status = migraphx_parse_onnx(&p, "conv_relu_maxpool_test.onnx", onnx_options);
    EXPECT(status == migraphx_status_success);

    std::string filename = "migraphx_c_api_netron_output_test.onnx";
    status               = migraphx_program_write_netron_output(p, filename.c_str());
    EXPECT(status == migraphx_status_success);

    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    EXPECT(ifs.good());
    auto size = ifs.tellg();
    EXPECT(size > 0);

    std::remove(filename.c_str());
    migraphx_program_destroy(p);
    migraphx_onnx_options_destroy(onnx_options);
}

TEST_CASE(netron_output_constructed_program)
{
    migraphx::program p;
    migraphx::module m = p.get_main_module();
    migraphx::shape s{migraphx_shape_float_type, {2, 3}};
    auto x      = m.add_parameter("x", s);
    auto y      = m.add_parameter("y", s);
    auto add_op = migraphx::operation("add");
    auto r      = m.add_instruction(add_op, {x, y});
    m.add_return({r});

    std::string filename = "migraphx_api_netron_constructed_test.onnx";
    p.write_netron_output(filename.c_str());

    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    EXPECT(ifs.good());
    auto size = ifs.tellg();
    EXPECT(size > 0);

    std::remove(filename.c_str());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
