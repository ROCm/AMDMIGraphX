/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(pack_int4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {0x0A, 0x0B, 0x0C, 0x0D}});
    mm->add_instruction(migraphx::make_op("pack_int4"), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0xBA), static_cast<uint8_t>(0xDC)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pack_int4_transposed)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {2, 2}, {1, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {0x0A, 0x0B, 0x0C, 0x0D}});
    mm->add_instruction(migraphx::make_op("pack_int4"), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0xBA), static_cast<uint8_t>(0xDC)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pack_int4_broadcasted)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {4}, {1}};
    auto l0  = mm->add_literal(migraphx::literal{s, {0x0A, 0x0B, 0x0C, 0x0D}});
    auto l0b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), l0);
    mm->add_instruction(migraphx::make_op("pack_int4"), l0b);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0xBA),
                              static_cast<uint8_t>(0xDC),
                              static_cast<uint8_t>(0xBA),
                              static_cast<uint8_t>(0xDC),
                              static_cast<uint8_t>(0xBA),
                              static_cast<uint8_t>(0xDC),
                              static_cast<uint8_t>(0xBA),
                              static_cast<uint8_t>(0xDC)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pack_int4_axis_0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {0x0A, 0x0B, 0x0C, 0x0D}});
    mm->add_instruction(migraphx::make_op("pack_int4", {{"axis", 0}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0xCA), static_cast<uint8_t>(0xDB)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pack_int4_nchw)
{
    // test with literal values such as 0x18 in which first 4 bits will be dropped, ideally
    // quantizer should produce values that fit into 4 bits.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {1, 2, 4, 4}};
    auto l0 = mm->add_literal(
        migraphx::literal{s, {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
                              0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
                              0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F}});
    mm->add_instruction(migraphx::make_op("pack_int4", {{"axis", -1}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0x10),
                              static_cast<uint8_t>(0x32),
                              static_cast<uint8_t>(0x54),
                              static_cast<uint8_t>(0x76),
                              static_cast<uint8_t>(0x98),
                              static_cast<uint8_t>(0xBA),
                              static_cast<uint8_t>(0xDC),
                              static_cast<uint8_t>(0xFE),
                              static_cast<uint8_t>(0x10),
                              static_cast<uint8_t>(0x32),
                              static_cast<uint8_t>(0x54),
                              static_cast<uint8_t>(0x76),
                              static_cast<uint8_t>(0x98),
                              static_cast<uint8_t>(0xBA),
                              static_cast<uint8_t>(0xDC),
                              static_cast<uint8_t>(0xFE)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
