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
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m, std::set<migraphx::shape::type_t> types)
{
    migraphx::run_passes(
        m,
        {migraphx::eliminate_data_type{std::move(types), migraphx::shape::float_type},
         migraphx::eliminate_identity{},
         migraphx::dead_code_elimination{}});
}

static void run_pass_to_int32(migraphx::module& m, std::set<migraphx::shape::type_t> types)
{
    migraphx::run_passes(
        m,
        {migraphx::eliminate_data_type{std::move(types), migraphx::shape::int32_type},
         migraphx::eliminate_identity{},
         migraphx::dead_code_elimination{}});
}

TEST_CASE(simple)
{
    migraphx::shape s{migraphx::shape::int8_type, {2, 2}};
    migraphx::module mm1;
    {
        auto x = mm1.add_parameter("x", s);
        auto y = mm1.add_parameter("y", s);
        mm1.add_instruction(migraphx::make_op("add"), x, y);
    }
    run_pass(mm1, {migraphx::shape::int8_type});

    migraphx::module mm2;
    {
        auto x      = mm2.add_parameter("x", s);
        auto y      = mm2.add_parameter("y", s);
        auto floatx = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto floaty = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto add = mm2.add_instruction(migraphx::make_op("add"), floatx, floaty);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}), add);
    }
    EXPECT(mm1 == mm2);
}

TEST_CASE(quant)
{
    migraphx::shape s{migraphx::shape::int8_type, {2, 2}};
    migraphx::module mm1;
    {
        auto x = mm1.add_parameter("x", s);
        auto y = mm1.add_parameter("y", s);
        mm1.add_instruction(migraphx::make_op("quant_dot"), x, y);
    }
    run_pass(mm1, {migraphx::shape::int8_type});

    migraphx::module mm2;
    {
        auto x      = mm2.add_parameter("x", s);
        auto y      = mm2.add_parameter("y", s);
        auto floatx = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto floaty = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto add = mm2.add_instruction(migraphx::make_op("dot"), floatx, floaty);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), add);
    }
    EXPECT(mm1 == mm2);
}

// Test int64 to int32 conversion for mod operation
// This ensures integer semantics are preserved (not converted to float)
TEST_CASE(int64_to_int32_mod)
{
    migraphx::shape s{migraphx::shape::int64_type, {1, 96}};
    migraphx::module mm1;
    {
        auto x = mm1.add_parameter("x", s);
        auto y = mm1.add_parameter("y", s);
        mm1.add_instruction(migraphx::make_op("mod"), x, y);
    }
    run_pass_to_int32(mm1, {migraphx::shape::int64_type});

    migraphx::module mm2;
    {
        auto x     = mm2.add_parameter("x", s);
        auto y     = mm2.add_parameter("y", s);
        auto int32x = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), x);
        auto int32y = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), y);
        auto mod_result = mm2.add_instruction(migraphx::make_op("mod"), int32x, int32y);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), mod_result);
    }
    EXPECT(mm1 == mm2);
}

// Test int64 to int32 conversion for mul then mod (common pattern in gather index computation)
TEST_CASE(int64_to_int32_mul_mod)
{
    migraphx::shape s{migraphx::shape::int64_type, {1, 96}};
    migraphx::module mm1;
    {
        auto x = mm1.add_parameter("x", s);
        auto y = mm1.add_parameter("y", s);
        auto z = mm1.add_parameter("z", s);
        auto mul = mm1.add_instruction(migraphx::make_op("mul"), x, y);
        mm1.add_instruction(migraphx::make_op("mod"), mul, z);
    }
    run_pass_to_int32(mm1, {migraphx::shape::int64_type});

    migraphx::module mm2;
    {
        auto x      = mm2.add_parameter("x", s);
        auto y      = mm2.add_parameter("y", s);
        auto z      = mm2.add_parameter("z", s);
        auto int32x = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), x);
        auto int32y = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), y);
        auto int32z = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), z);
        auto mul = mm2.add_instruction(migraphx::make_op("mul"), int32x, int32y);
        auto mul_back = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), mul);
        auto mul_to_int32 = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), mul_back);
        auto mod_result = mm2.add_instruction(migraphx::make_op("mod"), mul_to_int32, int32z);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), mod_result);
    }
    EXPECT(mm1 == mm2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
