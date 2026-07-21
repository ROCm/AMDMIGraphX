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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <test.hpp>

TEST_CASE(resolve_sym_expr_single_symbol)
{
    // Evaluate two exprs (n and floor(n/2)) of one root symbol n from its runtime value.
    auto n    = migraphx::sym::var("n", {1, 16});
    auto half = n / migraphx::sym::lit(2);

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sv_shape{migraphx::shape::int64_type, {1}};
    auto sv = mm->add_parameter("sym_vals", sv_shape);
    mm->add_instruction(
        migraphx::make_op(
            "resolve_sym_expr",
            {{"exprs", migraphx::value::array{migraphx::to_value(n), migraphx::to_value(half)}},
             {"symbols", migraphx::value::array{migraphx::to_value(n)}}}),
        sv);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> sv_data = {7};
    params["sym_vals"]           = migraphx::argument(sv_shape, sv_data.data());
    auto result                  = p.eval(params).back();

    // Output is a tuple: element i = eval(exprs[i]). n = 7, floor(7 / 2) = 3.
    migraphx::shape elem{migraphx::shape::int64_type, {1}};
    EXPECT(result.get_shape() == migraphx::shape{std::vector<migraphx::shape>{elem, elem}});
    auto subs = result.get_sub_objects();
    EXPECT(subs.size() == 2);
    EXPECT(subs[0].at<int64_t>() == 7);
    EXPECT(subs[1].at<int64_t>() == 3);
}

TEST_CASE(resolve_sym_expr_multi_symbol)
{
    // Two root symbols; one scalar value input per symbol, in `symbols` order.
    auto m   = migraphx::sym::var("m", {1, 16});
    auto n   = migraphx::sym::var("n", {1, 16});
    auto sum = m + n;

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape val{migraphx::shape::int64_type, {1}};
    auto mv = mm->add_parameter("m_val", val);
    auto nv = mm->add_parameter("n_val", val);
    mm->add_instruction(
        migraphx::make_op(
            "resolve_sym_expr",
            {{"exprs", migraphx::value::array{migraphx::to_value(sum)}},
             {"symbols", migraphx::value::array{migraphx::to_value(m), migraphx::to_value(n)}}}),
        mv,
        nv);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> m_data = {3};
    std::vector<int64_t> n_data = {4};
    params["m_val"]             = migraphx::argument(val, m_data.data());
    params["n_val"]             = migraphx::argument(val, n_data.data());
    auto result                 = p.eval(params).back();

    // Single-element tuple: m + n = 3 + 4.
    auto subs = result.get_sub_objects();
    EXPECT(subs.size() == 1);
    EXPECT(subs[0].at<int64_t>() == 7);
}
