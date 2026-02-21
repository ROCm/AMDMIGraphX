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

#include <op_builder_test_utils.hpp>

#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

namespace {
std::vector<float> run_rotary_embedding(migraphx::module m,
                                        const migraphx::shape& in_shape,
                                        const migraphx::shape& cache_shape,
                                        std::vector<float> in_data,
                                        std::vector<float> cos_data,
                                        std::vector<float> sin_data)
{
    migraphx::program p{std::move(m)};
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_data.data());

    migraphx::argument result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}
} // namespace

TEST_CASE(rotary_embedding_non_interleaved_structure_test)
{
    migraphx::module mm;
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 2, 1, 4}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 1, 1, 4}};

    auto input = mm.add_parameter("input", in_shape);
    auto cos   = mm.add_parameter("cos_cache", cache_shape);
    auto sin   = mm.add_parameter("sin_cache", cache_shape);

    migraphx::op::builder::add("rotary_embedding", mm, {input, cos, sin}, {{"interleaved", false}});

    migraphx::module expected;
    auto e_input = expected.add_parameter("input", in_shape);
    auto e_cos   = expected.add_parameter("cos_cache", cache_shape);
    auto e_sin   = expected.add_parameter("sin_cache", cache_shape);

    auto signs = expected.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {-1.0f, 1.0f}});
    signs = expected.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1}}}), signs);
    signs = expected.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}),
                                     signs);
    signs = expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), signs);

    auto first_half = expected.add_instruction(
        migraphx::make_op("slice", {{"axes", {-1}}, {"starts", {0}}, {"ends", {2}}}), e_input);
    auto second_half = expected.add_instruction(
        migraphx::make_op("slice", {{"axes", {-1}}, {"starts", {2}}, {"ends", {4}}}), e_input);
    auto rotated = expected.add_instruction(
        migraphx::make_op("concat", {{"axis", -1}}), second_half, first_half);

    signs = expected.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 1, 4}}}), signs);

    auto mul_cos = add_common_op(expected, migraphx::make_op("mul"), {e_input, e_cos});
    auto mul_sin = add_common_op(expected, migraphx::make_op("mul"), {signs, e_sin});
    mul_sin      = add_common_op(expected, migraphx::make_op("mul"), {mul_sin, rotated});
    add_common_op(expected, migraphx::make_op("add"), {mul_cos, mul_sin});

    EXPECT(mm == expected);
}

TEST_CASE(rotary_embedding_interleaved_structure_test)
{
    migraphx::module mm;
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 2, 1, 4}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 1, 1, 4}};

    auto input = mm.add_parameter("input", in_shape);
    auto cos   = mm.add_parameter("cos_cache", cache_shape);
    auto sin   = mm.add_parameter("sin_cache", cache_shape);

    migraphx::op::builder::add("rotary_embedding", mm, {input, cos, sin}, {{"interleaved", true}});

    migraphx::module expected;
    auto e_input = expected.add_parameter("input", in_shape);
    auto e_cos   = expected.add_parameter("cos_cache", cache_shape);
    auto e_sin   = expected.add_parameter("sin_cache", cache_shape);

    auto signs = expected.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {2}}, {-1.0f, 1.0f}});
    signs = expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2}}}), signs);
    signs = expected.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}),
                                     signs);
    signs = expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), signs);

    auto rs_in =
        expected.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2}}}), e_input);
    auto evens = expected.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rs_in);
    auto odds = expected.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rs_in);
    auto swapped =
        expected.add_instruction(migraphx::make_op("concat", {{"axis", -1}}), odds, evens);
    auto rotated =
        expected.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 1, 4}}}), swapped);

    signs = expected.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 1, 4}}}), signs);

    auto mul_cos = add_common_op(expected, migraphx::make_op("mul"), {e_input, e_cos});
    auto mul_sin = add_common_op(expected, migraphx::make_op("mul"), {signs, e_sin});
    mul_sin      = add_common_op(expected, migraphx::make_op("mul"), {mul_sin, rotated});
    add_common_op(expected, migraphx::make_op("add"), {mul_cos, mul_sin});

    EXPECT(mm == expected);
}

TEST_CASE(rotary_embedding_verify_non_interleaved_test)
{
    // input: [1, 1, 1, 6] (batch=1, heads=1, seq=1, D=6)
    // cos, sin: [1, 1, 1, 6]
    // Non-interleaved: output[i] = in[i]*cos[i] + sign[i]*in[rotate(i)]*sin[i]
    // sign = [-1,-1,-1, 1,1,1], rotate swaps halves: [3,4,5, 0,1,2]
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 6}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 1, 1, 6}};

    migraphx::module mm;
    auto input = mm.add_parameter("input", in_shape);
    auto cos   = mm.add_parameter("cos_cache", cache_shape);
    auto sin   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add("rotary_embedding", mm, {input, cos, sin}, {{"interleaved", false}});

    std::vector<float> in_data  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> cos_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> sin_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // With cos=1, sin=0: output = input
    auto result = run_rotary_embedding(mm, in_shape, cache_shape, in_data, cos_data, sin_data);
    EXPECT(migraphx::verify::verify_rms_range(result, in_data));

    // With cos=0, sin=1: output[i] = sign[i] * input[rotate(i)]
    // signs = [-1,-1,-1, 1,1,1], rotated = [4,5,6, 1,2,3]
    // output = [-4,-5,-6, 1,2,3]
    cos_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    sin_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::module mm2;
    auto input2 = mm2.add_parameter("input", in_shape);
    auto cos2   = mm2.add_parameter("cos_cache", cache_shape);
    auto sin2   = mm2.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm2, {input2, cos2, sin2}, {{"interleaved", false}});

    result = run_rotary_embedding(mm2, in_shape, cache_shape, in_data, cos_data, sin_data);
    std::vector<float> expected = {-4.0f, -5.0f, -6.0f, 1.0f, 2.0f, 3.0f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(rotary_embedding_verify_interleaved_test)
{
    // input: [1, 1, 1, 4] (batch=1, heads=1, seq=1, D=4)
    // Interleaved: pairs (x,y) -> output = x*cos - y*sin, y*cos + x*sin
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 4}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 1, 1, 4}};

    migraphx::module mm;
    auto input = mm.add_parameter("input", in_shape);
    auto cos   = mm.add_parameter("cos_cache", cache_shape);
    auto sin   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add("rotary_embedding", mm, {input, cos, sin}, {{"interleaved", true}});

    // in = [1, 2, 3, 4], cos = [1,1,1,1], sin = [0,0,0,0]
    // output = input (rotation by 0)
    std::vector<float> in_data  = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> cos_data = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> sin_data = {0.0f, 0.0f, 0.0f, 0.0f};

    auto result = run_rotary_embedding(mm, in_shape, cache_shape, in_data, cos_data, sin_data);
    EXPECT(migraphx::verify::verify_rms_range(result, in_data));

    // With cos=0, sin=1: signs=[-1,1,-1,1], rotated=[2,1,4,3]
    // output = 0 + signs * rotated * sin = [-2,1,-4,3]
    cos_data = {0.0f, 0.0f, 0.0f, 0.0f};
    sin_data = {1.0f, 1.0f, 1.0f, 1.0f};

    migraphx::module mm2;
    auto input2 = mm2.add_parameter("input", in_shape);
    auto cos2   = mm2.add_parameter("cos_cache", cache_shape);
    auto sin2   = mm2.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm2, {input2, cos2, sin2}, {{"interleaved", true}});

    result = run_rotary_embedding(mm2, in_shape, cache_shape, in_data, cos_data, sin_data);
    std::vector<float> expected = {-2.0f, 1.0f, -4.0f, 3.0f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(rotary_embedding_verify_mixed_cos_sin_test)
{
    // Test with non-trivial cos/sin values (90-degree rotation)
    // cos=0, sin=1 applied to non-interleaved [1, 0, 0, 1] with D=4
    // signs = [-1,-1, 1,1], rotated = [0,1, 1,0]
    // output = [1,0,0,1]*0 + [-1,-1,1,1]*[0,1,1,0]*1 = [0,-1,1,0]
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 4}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 1, 1, 4}};

    migraphx::module mm;
    auto input = mm.add_parameter("input", in_shape);
    auto cos   = mm.add_parameter("cos_cache", cache_shape);
    auto sin   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add("rotary_embedding", mm, {input, cos, sin}, {{"interleaved", false}});

    std::vector<float> in_data  = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> cos_data = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> sin_data = {1.0f, 1.0f, 1.0f, 1.0f};

    auto result = run_rotary_embedding(mm, in_shape, cache_shape, in_data, cos_data, sin_data);
    std::vector<float> expected = {0.0f, -1.0f, 1.0f, 0.0f};
    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(rotary_embedding_4arg_cache_gather_verify_test)
{
    // 4-arg mode: builder gathers cos/sin from raw caches using position IDs
    // input: [1, 1, 1, 4] (batch=1, heads=1, seq=1, D=4)
    // pos_ids: [1, 1] = {0} -> gather row 0 from cache
    // cos_cache: [2, 2] (max_seq=2, half_head=2)
    // sin_cache: [2, 2]
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {1, 1}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {2, 2}};

    migraphx::module mm;
    auto input   = mm.add_parameter("input", in_shape);
    auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
    auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
    auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    // cos_cache row 0 = [1, 1] -> doubled = [1,1,1,1]
    // sin_cache row 0 = [0, 0] -> doubled = [0,0,0,0]
    // With cos=1, sin=0: output = input
    std::vector<float> in_data        = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> pos_data         = {0};
    std::vector<float> cos_cache_data = {1.0f, 1.0f, 0.5f, 0.5f};
    std::vector<float> sin_cache_data = {0.0f, 0.0f, 0.8f, 0.8f};

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Position 0: cos=[1,1,1,1], sin=[0,0,0,0] -> output = input
    EXPECT(migraphx::verify::verify_rms_range(result_vector, in_data));

    // Now gather from position 1: cos=[0.5,0.5,0.5,0.5], sin=[0.8,0.8,0.8,0.8]
    // signs = [-1,-1,1,1], rotated = [3,4,1,2]
    // output[i] = in[i]*0.5 + signs[i]*rotated[i]*0.8
    // = [1*0.5+(-1)*3*0.8, 2*0.5+(-1)*4*0.8, 3*0.5+1*1*0.8, 4*0.5+1*2*0.8]
    // = [0.5-2.4, 1.0-3.2, 1.5+0.8, 2.0+1.6] = [-1.9, -2.2, 2.3, 3.6]
    pos_data      = {1};
    pp["pos_ids"] = migraphx::argument(pos_shape, pos_data.data());

    result = p.eval(pp).back();
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> expected = {-1.9f, -2.2f, 2.3f, 3.6f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
}
