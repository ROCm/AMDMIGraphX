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

#include <onnx_test.hpp>
#include <migraphx/float8.hpp>

TEST_CASE(constant_fp8e4m3fn_int32_data_test)
{
    using migraphx::fp8::fp8e4m3fn;
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::fp8e4m3fn_type, {3}},
                                      std::vector<fp8e4m3fn>{
                                          fp8e4m3fn{0x38, fp8e4m3fn::from_bits()},
                                          fp8e4m3fn{0xB8, fp8e4m3fn::from_bits()},
                                          fp8e4m3fn{0x01, fp8e4m3fn::from_bits()},
                                      }});
    auto prog = optimize_onnx("constant_fp8e4m3fn_int32_data_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fp8e5m2_int32_data_test)
{
    using migraphx::fp8::fp8e5m2;
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::fp8e5m2_type, {3}},
                                      std::vector<fp8e5m2>{
                                          fp8e5m2{0x38, fp8e5m2::from_bits()},
                                          fp8e5m2{0xB8, fp8e5m2::from_bits()},
                                          fp8e5m2{0x01, fp8e5m2::from_bits()},
                                      }});
    auto prog = optimize_onnx("constant_fp8e5m2_int32_data_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(constant_fp8e4m3fnuz_int32_data_test)
{
    using migraphx::fp8::fp8e4m3fnuz;
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, {3}},
                                      std::vector<fp8e4m3fnuz>{
                                          fp8e4m3fnuz{0x38, fp8e4m3fnuz::from_bits()},
                                          fp8e4m3fnuz{0xB8, fp8e4m3fnuz::from_bits()},
                                          fp8e4m3fnuz{0x01, fp8e4m3fnuz::from_bits()},
                                      }});
    auto prog = optimize_onnx("constant_fp8e4m3fnuz_int32_data_test.onnx");

    EXPECT(p == prog);
}
