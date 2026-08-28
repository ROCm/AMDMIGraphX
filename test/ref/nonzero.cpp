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

#include <test.hpp>

TEST_CASE(nonzero_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> data = {
        1.0f, 1.3f, 0.0f, -1.2f, 0.0f, -100.f, 200.f, 0.0f, 0.1f, 0.2f, 0.0f, 0.5f};
    auto input = mm->add_literal(migraphx::literal(s, data));
    auto nz    = mm->add_instruction(migraphx::make_op("nonzero"), input);
    auto ret   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nz);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                 1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(nonzero_num_nonzero)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}};
    std::vector<float> data = {
        1.0f, 1.3f, 0.0f, -1.2f, 0.0f, -100.f, 200.f, 0.0f, 0.1f, 0.2f, 0.0f, 0.5f};
    auto input       = mm->add_literal(migraphx::literal(s, data));
    auto nz          = mm->add_instruction(migraphx::make_op("nonzero"), input);
    auto num_nonzero = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nz);
    mm->add_return({num_nonzero});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(result_vector == std::vector<int64_t>{8});
}

TEST_CASE(nonzero_dyn_input)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {3, 3}}};
    auto input       = mm->add_parameter("data", s);
    auto nz          = mm->add_instruction(migraphx::make_op("nonzero"), input);
    auto indices     = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nz);
    auto num_nonzero = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nz);
    mm->add_return({indices, num_nonzero});
    p.compile(migraphx::make_target("ref"));

    // Run with 2 of the 4 possible rows.
    migraphx::shape input_s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f};
    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(input_s, data.data());

    auto results = p.eval(pp);

    std::vector<int64_t> indices_vector;
    results.at(0).visit([&](auto output) { indices_vector.assign(output.begin(), output.end()); });
    // np.nonzero(data.reshape(2, 3)) is ((0, 0, 1, 1), (0, 2, 1, 2)). The buffer stays sized for
    // the 4x3 maximum, so each of the two rows carries 8 columns of padding.
    std::vector<int64_t> gold = {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(indices_vector == gold);

    std::vector<int64_t> num_nonzero_vector;
    results.at(1).visit(
        [&](auto output) { num_nonzero_vector.assign(output.begin(), output.end()); });
    EXPECT(num_nonzero_vector == std::vector<int64_t>{4});
}

TEST_CASE(nonzero_transposed_input)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 4.0f};
    auto input              = mm->add_literal(migraphx::literal(s, data));
    auto transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), input);
    auto nz  = mm->add_instruction(migraphx::make_op("nonzero"), transposed);
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nz);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    // np.nonzero(data.reshape(2, 3).T), padded to nonzero output shape {2, 6}.
    std::vector<int64_t> gold = {0, 1, 2, 2, 0, 0, 0, 1, 0, 1, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(nonzero_broadcasted_input)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {1.0f}});
    auto broadcasted =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), input);
    auto nz  = mm->add_instruction(migraphx::make_op("nonzero"), broadcasted);
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nz);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int64_t> gold = {0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
