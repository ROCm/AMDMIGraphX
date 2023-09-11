/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(topk_test)
{
    auto create_program = [](int64_t k, int64_t axis, int largest) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        auto data = mm->add_parameter("data", s);
        auto r    = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", axis}, {"k", k}, {"largest", largest}}), data);
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        mm->add_return({r0, r1});

        return p;
    };

    auto run_program = [&](int64_t k, int64_t axis, int largest) {
        auto p = create_program(k, axis, largest);
        p.compile(migraphx::make_target("ref"));
        std::vector<float> data = {
            2.1, 2.3, 2.0, 2.5, 1.9, 3.3, 0.2, 4.5, 0.1, 0.8, 1.0, 4.5, 2.1, 0.8, 1.5};
        migraphx::shape s{migraphx::shape::float_type, {3, 5}};
        migraphx::parameter_map pp;
        pp["data"] = migraphx::argument(s, data.data());
        auto rets  = p.eval(pp);
        std::vector<float> ret_val;
        rets.front().visit([&](auto v) { ret_val.assign(v.begin(), v.end()); });
        std::vector<int64_t> ret_ind;
        rets.back().visit([&](auto v) { ret_ind.assign(v.begin(), v.end()); });

        return std::make_pair(ret_val, ret_ind);
    };

    // case 1
    {
        auto results                = run_program(4, 1, 1);
        std::vector<float> gold_val = {2.5, 2.3, 2.1, 2, 4.5, 3.3, 0.8, 0.2, 4.5, 2.1, 1.5, 1};
        EXPECT(results.first == gold_val);
        std::vector<int64_t> gold_ind = {3, 1, 0, 2, 2, 0, 4, 1, 1, 2, 4, 0};
        EXPECT(results.second == gold_ind);
    }

    // case 2
    {
        auto results                = run_program(4, 1, 0);
        std::vector<float> gold_val = {1.9, 2, 2.1, 2.3, 0.1, 0.2, 0.8, 3.3, 0.8, 1, 1.5, 2.1};
        EXPECT(results.first == gold_val);
        std::vector<int64_t> gold_ind = {4, 2, 0, 1, 3, 1, 4, 0, 3, 0, 4, 2};
        EXPECT(results.second == gold_ind);
    }
}
