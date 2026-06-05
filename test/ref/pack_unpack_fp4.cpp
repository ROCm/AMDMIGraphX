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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/float_equal.hpp>

#include <test.hpp>

TEST_CASE(pack_fp4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {-2.f, 3.4f, 3.5f, 0.f}});
    mm->add_instruction(migraphx::make_op("pack_fp4", {{"axis", 1}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    result =
        result.reshape(migraphx::shape(migraphx::shape::uint8_type, result.get_shape().lens()));
    std::vector<uint8_t> results_vector(2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold{0x5C, 0x06};
    EXPECT(results_vector.at(0) == gold.at(0));
    EXPECT(results_vector.at(1) == gold.at(1));
}

TEST_CASE(unpack_fp4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::fp4x2_type, {2, 1}};
    std::vector<uint8_t> packed_data = {0x5C, 0x06};
    auto lit                         = migraphx::literal{s, packed_data.data()};
    auto l0                          = mm->add_literal(lit);
    mm->add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 1}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-2.f, 3.f, 4.f, 0.f};
    EXPECT(migraphx::float_equal(results_vector.at(0), gold.at(0)));
    EXPECT(migraphx::float_equal(results_vector.at(1), gold.at(1)));
    EXPECT(migraphx::float_equal(results_vector.at(2), gold.at(2)));
    EXPECT(migraphx::float_equal(results_vector.at(3), gold.at(3)));
}

TEST_CASE(pack_unpack_fp4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0       = mm->add_literal(migraphx::literal{s, {-2.f, 3.4f, 3.5f, 0.f}});
    auto pack_ins = mm->add_instruction(migraphx::make_op("pack_fp4", {{"axis", 1}}), l0);
    mm->add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 1}}), pack_ins);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(4);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-2.f, 3.f, 4.f, 0.f};
    EXPECT(migraphx::float_equal(results_vector.at(0), gold.at(0)));
    EXPECT(migraphx::float_equal(results_vector.at(1), gold.at(1)));
    EXPECT(migraphx::float_equal(results_vector.at(2), gold.at(2)));
    EXPECT(migraphx::float_equal(results_vector.at(3), gold.at(3)));
}
