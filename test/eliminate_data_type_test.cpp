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
#include <string>

static void run_pass(migraphx::module& m, std::set<migraphx::shape::type_t> types)
{
    migraphx::run_passes(
        m,
        {migraphx::eliminate_data_type{std::move(types), migraphx::shape::float_type},
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

static void check_skip_slice(const std::string& name)
{
    migraphx::shape data_shape{migraphx::shape::int64_type, {4}};
    migraphx::shape index_shape{migraphx::shape::int64_type, {1}};
    auto slice_op = name == "slice"
                        ? migraphx::make_op(name, {{"axes", {0}}})
                        : migraphx::make_op(name, {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}});

    migraphx::module mm1;
    {
        auto data   = mm1.add_parameter("data", data_shape);
        auto starts = mm1.add_parameter("starts", index_shape);
        auto ends   = mm1.add_parameter("ends", index_shape);
        auto slice  = mm1.add_instruction(slice_op, data, starts, ends);
        mm1.add_instruction(migraphx::make_op("relu"), slice);
    }
    run_pass(mm1, {migraphx::shape::int64_type});

    migraphx::module mm2;
    {
        auto data      = mm2.add_parameter("data", data_shape);
        auto starts    = mm2.add_parameter("starts", index_shape);
        auto ends      = mm2.add_parameter("ends", index_shape);
        auto slice     = mm2.add_instruction(slice_op, data, starts, ends);
        auto converted = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), slice);
        auto relu = mm2.add_instruction(migraphx::make_op("relu"), converted);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), relu);
    }

    EXPECT(mm1 == mm2);
}

TEST_CASE(skip_slice) { check_skip_slice("slice"); }

TEST_CASE(skip_dyn_slice) { check_skip_slice("dyn_slice"); }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
