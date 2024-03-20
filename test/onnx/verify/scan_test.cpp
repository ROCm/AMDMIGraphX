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

#include "migraphx/argument.hpp"
#include "migraphx/generate.hpp"
#include "migraphx/module.hpp"
#include "migraphx/onnx.hpp"
#include "migraphx/shape.hpp"
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

static migraphx::shape make_shape(const std::vector<size_t>& lens)
{
    return migraphx::shape{migraphx::shape::float_type, lens};
}

static std::vector<float> arg_to_vec(const migraphx::argument& arg)
{
    std::vector<float> ret;
    arg.visit([&](auto output) { ret.assign(output.begin(), output.end()); });
    return ret;
}

TEST_CASE(scan_test1)
{
    auto prog = migraphx::parse_onnx("scan_test1.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(4, 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {3, 2, 2}};
    std::vector<float> scan_ins(12);
    std::iota(scan_ins.begin(), scan_ins.end(), 1);
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 3);

    auto final_state = result[0];
    auto scan_out1   = result[1];
    auto scan_out2   = result[2];

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{15, 18, 21, 24};
    EXPECT(arg_to_vec(final_state) == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24};
    EXPECT(arg_to_vec(scan_out1) == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{4, 6, 16, 20, 36, 42};
    EXPECT(arg_to_vec(scan_out2) == scan_out2_gold);
}

TEST_CASE(scan_test2)
{
    auto prog = migraphx::parse_onnx("scan_test2.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(4, 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {3, 2, 2}};
    std::vector<float> scan_ins(12);
    std::iota(scan_ins.begin(), scan_ins.end(), 1);
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 3);

    auto final_state = result[0];
    auto scan_out1   = result[1];
    auto scan_out2   = result[2];

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{15, 18, 21, 24};
    EXPECT(arg_to_vec(final_state) == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{15, 18, 21, 24, 6, 8, 10, 12, 1, 2, 3, 4};
    EXPECT(arg_to_vec(scan_out1) == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{4, 6, 16, 20, 36, 42};
    EXPECT(arg_to_vec(scan_out2) == scan_out2_gold);
}

TEST_CASE(scan_test3)
{
    auto prog = migraphx::parse_onnx("scan_test3.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(4, 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {3, 2, 2}};
    std::vector<float> scan_ins(12);
    std::iota(scan_ins.begin(), scan_ins.end(), 1);
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 3);

    auto final_state = result[0];
    auto scan_out1   = result[1];
    auto scan_out2   = result[2];

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{15, 18, 21, 24};
    EXPECT(arg_to_vec(final_state) == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({2, 3, 2}));
    std::vector<float> scan_out1_gold{1, 2, 6, 8, 15, 18, 3, 4, 10, 12, 21, 24};
    EXPECT(arg_to_vec(scan_out1) == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({2, 3}));
    std::vector<float> scan_out2_gold{36, 16, 4, 42, 20, 6};
    EXPECT(arg_to_vec(scan_out2) == scan_out2_gold);
}

TEST_CASE(scan_test4)
{
    auto prog = migraphx::parse_onnx("scan_test4.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(4, 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {3, 2, 2}};
    std::vector<float> scan_ins(12);
    std::iota(scan_ins.begin(), scan_ins.end(), 1);
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 3);

    auto final_state = result[0];
    auto scan_out1   = result[1];
    auto scan_out2   = result[2];

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{15, 18, 21, 24};
    EXPECT(arg_to_vec(final_state) == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{9, 10, 11, 12, 14, 16, 18, 20, 15, 18, 21, 24};
    EXPECT(arg_to_vec(scan_out1) == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{20, 22, 32, 36, 36, 42};
    EXPECT(arg_to_vec(scan_out2) == scan_out2_gold);
}

TEST_CASE(scan_test5)
{
    auto prog = migraphx::parse_onnx("scan_test5.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2, 2}};
    std::vector<float> init_state(4, 0);
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {2, 3, 2}};
    std::vector<float> scan_ins(12);
    std::iota(scan_ins.begin(), scan_ins.end(), 1);
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result = prog.eval(pm);
    EXPECT(result.size() == 3);

    auto final_state = result[0];
    auto scan_out1   = result[1];
    auto scan_out2   = result[2];

    EXPECT(final_state.get_shape() == make_shape({2, 2}));
    std::vector<float> final_state_gold{9, 12, 27, 30};
    EXPECT(arg_to_vec(final_state) == final_state_gold);

    EXPECT(scan_out1.get_shape() == make_shape({3, 2, 2}));
    std::vector<float> scan_out1_gold{1, 2, 7, 8, 4, 6, 16, 18, 9, 12, 27, 30};
    EXPECT(arg_to_vec(scan_out1) == scan_out1_gold);

    EXPECT(scan_out2.get_shape() == make_shape({3, 2}));
    std::vector<float> scan_out2_gold{8, 10, 20, 24, 36, 42};
    EXPECT(arg_to_vec(scan_out2) == scan_out2_gold);
}
