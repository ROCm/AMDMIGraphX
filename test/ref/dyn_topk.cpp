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
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

using migraphx::sym::var;

static const migraphx::shape data_shape{migraphx::shape::float_type, {2, 4}};
static const std::vector<float> data_values = {1, 3, 2, 4, 8, 5, 7, 6};

static migraphx::program make_dyn_topk_program(bool largest)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_literal(migraphx::literal{data_shape, data_values});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto out  = mm->add_instruction(
        migraphx::make_op(
            "dyn_topk",
            {{"k", migraphx::to_value(var("k", {1, 4}))}, {"axis", 1}, {"largest", largest}}),
        data,
        k);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    mm->add_return({val, ind});
    p.compile(migraphx::make_target("ref"));
    return p;
}

static std::pair<std::vector<float>, std::vector<int64_t>> run_with_k(migraphx::program& p,
                                                                      int64_t k)
{
    migraphx::shape ks{migraphx::shape::int64_type, {1}};
    std::vector<int64_t> k_data = {k};
    migraphx::parameter_map pp;
    pp["k"] = migraphx::argument(ks, k_data.data());

    auto results = p.eval(pp);
    std::vector<float> val;
    std::vector<int64_t> ind;
    results[0].visit([&](auto o) { val.assign(o.begin(), o.end()); });
    results[1].visit([&](auto o) { ind.assign(o.begin(), o.end()); });
    return {val, ind};
}

// The compile-time shape is symbolic, so the output size has to come from the `k` argument at
// eval time. Running one compiled program with two different `k` values proves it does.
TEST_CASE(dyn_topk_follows_runtime_k)
{
    auto p = make_dyn_topk_program(true);

    auto [val2, ind2] = run_with_k(p, 2);
    EXPECT(val2 == std::vector<float>{4, 3, 8, 7});
    EXPECT(ind2 == std::vector<int64_t>{3, 1, 0, 2});

    auto [val3, ind3] = run_with_k(p, 3);
    EXPECT(val3 == std::vector<float>{4, 3, 2, 8, 7, 6});
    EXPECT(ind3 == std::vector<int64_t>{3, 1, 2, 0, 2, 3});
}

TEST_CASE(dyn_topk_smallest)
{
    auto p          = make_dyn_topk_program(false);
    auto [val, ind] = run_with_k(p, 2);
    EXPECT(val == std::vector<float>{1, 2, 5, 6});
    EXPECT(ind == std::vector<int64_t>{0, 2, 1, 3});
}

// `k` at the axis length degenerates to a full sort.
TEST_CASE(dyn_topk_k_equals_axis)
{
    auto p          = make_dyn_topk_program(true);
    auto [val, ind] = run_with_k(p, 4);
    EXPECT(val == std::vector<float>{4, 3, 2, 1, 8, 7, 6, 5});
    EXPECT(ind == std::vector<int64_t>{3, 1, 2, 0, 0, 2, 3, 1});
}
