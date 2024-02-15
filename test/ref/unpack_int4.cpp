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

TEST_CASE(unpack_int4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {2, 1}};
    auto l0 = mm->add_literal(migraphx::literal{s, {0xBA, 0xDC}});
    mm->add_instruction(migraphx::make_op("unpack_int4"), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0x0A),
                              static_cast<uint8_t>(0x0B),
                              static_cast<uint8_t>(0x0C),
                              static_cast<uint8_t>(0x0D)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(unpack_int4_transposed)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {2, 2}, {1, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {0x1A, 0x2B, 0x3C, 0x4D}});
    mm->add_instruction(migraphx::make_op("unpack_int4"), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(8);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0x0A),
                              static_cast<uint8_t>(0x01),
                              static_cast<uint8_t>(0x0B),
                              static_cast<uint8_t>(0x02),
                              static_cast<uint8_t>(0x0C),
                              static_cast<uint8_t>(0x03),
                              static_cast<uint8_t>(0x0D),
                              static_cast<uint8_t>(0x04)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(unpack_int4_broadcasted)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {4}, {1}};
    auto l0  = mm->add_literal(migraphx::literal{s, {0x1A, 0x2B, 0x3C, 0x4D}});
    auto l0b = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), l0);
    mm->add_instruction(migraphx::make_op("unpack_int4"), l0b);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(32);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{
        static_cast<uint8_t>(0x0A), static_cast<uint8_t>(0x01), static_cast<uint8_t>(0x0B),
        static_cast<uint8_t>(0x02), static_cast<uint8_t>(0x0C), static_cast<uint8_t>(0x03),
        static_cast<uint8_t>(0x0D), static_cast<uint8_t>(0x04), static_cast<uint8_t>(0x0A),
        static_cast<uint8_t>(0x01), static_cast<uint8_t>(0x0B), static_cast<uint8_t>(0x02),
        static_cast<uint8_t>(0x0C), static_cast<uint8_t>(0x03), static_cast<uint8_t>(0x0D),
        static_cast<uint8_t>(0x04), static_cast<uint8_t>(0x0A), static_cast<uint8_t>(0x01),
        static_cast<uint8_t>(0x0B), static_cast<uint8_t>(0x02), static_cast<uint8_t>(0x0C),
        static_cast<uint8_t>(0x03), static_cast<uint8_t>(0x0D), static_cast<uint8_t>(0x04),
        static_cast<uint8_t>(0x0A), static_cast<uint8_t>(0x01), static_cast<uint8_t>(0x0B),
        static_cast<uint8_t>(0x02), static_cast<uint8_t>(0x0C), static_cast<uint8_t>(0x03),
        static_cast<uint8_t>(0x0D), static_cast<uint8_t>(0x04)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(unpack_int4_axis_0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {1, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {0xCA, 0xDB}});
    mm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 0}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{static_cast<uint8_t>(0x0A),
                              static_cast<uint8_t>(0x0B),
                              static_cast<uint8_t>(0x0C),
                              static_cast<uint8_t>(0x0D)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(unpack_int4_nchw)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::uint8_type, {1, 2, 4, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s,
                                                {0x10,
                                                 0x32,
                                                 0x54,
                                                 0x76,
                                                 0x98,
                                                 0xBA,
                                                 0xDC,
                                                 0xFE,
                                                 0x10,
                                                 0x32,
                                                 0x54,
                                                 0x76,
                                                 0x98,
                                                 0xBA,
                                                 0xDC,
                                                 0xFE}});
    mm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", -1}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<uint8_t> results_vector(32);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{
        static_cast<uint8_t>(0x00), static_cast<uint8_t>(0x01), static_cast<uint8_t>(0x02),
        static_cast<uint8_t>(0x03), static_cast<uint8_t>(0x04), static_cast<uint8_t>(0x05),
        static_cast<uint8_t>(0x06), static_cast<uint8_t>(0x07), static_cast<uint8_t>(0x08),
        static_cast<uint8_t>(0x09), static_cast<uint8_t>(0x0A), static_cast<uint8_t>(0x0B),
        static_cast<uint8_t>(0x0C), static_cast<uint8_t>(0x0D), static_cast<uint8_t>(0x0E),
        static_cast<uint8_t>(0x0F), static_cast<uint8_t>(0x00), static_cast<uint8_t>(0x01),
        static_cast<uint8_t>(0x02), static_cast<uint8_t>(0x03), static_cast<uint8_t>(0x04),
        static_cast<uint8_t>(0x05), static_cast<uint8_t>(0x06), static_cast<uint8_t>(0x07),
        static_cast<uint8_t>(0x08), static_cast<uint8_t>(0x09), static_cast<uint8_t>(0x0A),
        static_cast<uint8_t>(0x0B), static_cast<uint8_t>(0x0C), static_cast<uint8_t>(0x0D),
        static_cast<uint8_t>(0x0E), static_cast<uint8_t>(0x0F)};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
