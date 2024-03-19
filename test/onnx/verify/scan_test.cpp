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

TEST_CASE(scan_test)
{
    auto prog = migraphx::parse_onnx("scan_test.onnx");
    prog.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;

    migraphx::shape init_state_sh{migraphx::shape::float_type, {2}};
    std::vector<float> init_state{0, 0};
    pm["init_state"] = migraphx::argument(init_state_sh, init_state.data());

    migraphx::shape scan_ins_sh{migraphx::shape::float_type, {3, 2}};
    std::vector<float> scan_ins{1, 2, 3, 4, 5, 6};
    pm["scan_ins"] = migraphx::argument(scan_ins_sh, scan_ins.data());

    auto result      = prog.eval(pm);
    auto final_state = result[0];
    auto scan_out    = result[1];

    EXPECT(final_state.get_shape() == make_shape({2}));
    std::vector final_state_gold{9.f, 12.f};
    EXPECT(arg_to_vec(final_state) == final_state_gold);

    EXPECT(scan_out.get_shape() == make_shape({3, 2}));
    std::vector scan_out_gold{1.f, 2.f, 4.f, 6.f, 9.f, 12.f};
    EXPECT(arg_to_vec(scan_out) == scan_out_gold);
}
