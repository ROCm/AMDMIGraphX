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

TEST_CASE(rotary_embedding_verify_non_interleaved_test)
{
    // input: [1, 1, 1, 6] (batch=1, heads=1, seq=1, D=6)
    // cache: [1, 3] (max_seq=1, half_head=3), pos_ids={0}
    // Non-interleaved: output[i] = in[i]*cos[i] + sign[i]*in[rotate(i)]*sin[i]
    // sign = [-1,-1,-1, 1,1,1], rotate swaps halves: [3,4,5, 0,1,2]
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 6}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 3}};

    {
        migraphx::module mm;
        auto input   = mm.add_parameter("input", in_shape);
        auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
        auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
        auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
        migraphx::op::builder::add(
            "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

        migraphx::program p{std::move(mm)};
        p.compile(migraphx::make_target("ref"));

        std::vector<float> in_data        = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int> pos_data         = {0};
        std::vector<float> cos_cache_data = {1.0f, 1.0f, 1.0f};
        std::vector<float> sin_cache_data = {0.0f, 0.0f, 0.0f};

        migraphx::parameter_map pp;
        pp["input"]     = migraphx::argument(in_shape, in_data.data());
        pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
        pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
        pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_rms_range(result_vector, in_data));
    }

    {
        migraphx::module mm;
        auto input   = mm.add_parameter("input", in_shape);
        auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
        auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
        auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
        migraphx::op::builder::add(
            "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

        migraphx::program p{std::move(mm)};
        p.compile(migraphx::make_target("ref"));

        // With cos=0, sin=1: output[i] = sign[i] * input[rotate(i)]
        // signs = [-1,-1,-1, 1,1,1], rotated = [4,5,6, 1,2,3]
        // output = [-4,-5,-6, 1,2,3]
        std::vector<float> in_data        = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<int> pos_data         = {0};
        std::vector<float> cos_cache_data = {0.0f, 0.0f, 0.0f};
        std::vector<float> sin_cache_data = {1.0f, 1.0f, 1.0f};

        migraphx::parameter_map pp;
        pp["input"]     = migraphx::argument(in_shape, in_data.data());
        pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
        pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
        pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
        std::vector<float> expected = {-4.0f, -5.0f, -6.0f, 1.0f, 2.0f, 3.0f};
        EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
    }
}

TEST_CASE(rotary_embedding_verify_interleaved_test)
{
    // input: [1, 1, 1, 4] (batch=1, heads=1, seq=1, D=4)
    // cache: [1, 2] (max_seq=1, half_head=2), pos_ids={0}
    // Interleaved: pairs (x,y) -> output = x*cos - y*sin, y*cos + x*sin
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 2}};

    {
        migraphx::module mm;
        auto input   = mm.add_parameter("input", in_shape);
        auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
        auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
        auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
        migraphx::op::builder::add(
            "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", true}});

        migraphx::program p{std::move(mm)};
        p.compile(migraphx::make_target("ref"));

        // in = [1, 2, 3, 4], cos = [1,1,1,1], sin = [0,0,0,0]
        // output = input (rotation by 0)
        std::vector<float> in_data        = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int> pos_data         = {0};
        std::vector<float> cos_cache_data = {1.0f, 1.0f};
        std::vector<float> sin_cache_data = {0.0f, 0.0f};

        migraphx::parameter_map pp;
        pp["input"]     = migraphx::argument(in_shape, in_data.data());
        pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
        pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
        pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
        EXPECT(migraphx::verify::verify_rms_range(result_vector, in_data));
    }

    {
        migraphx::module mm;
        auto input   = mm.add_parameter("input", in_shape);
        auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
        auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
        auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
        migraphx::op::builder::add(
            "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", true}});

        migraphx::program p{std::move(mm)};
        p.compile(migraphx::make_target("ref"));

        // With cos=0, sin=1: signs=[-1,1,-1,1], rotated=[2,1,4,3]
        // output = 0 + signs * rotated * sin = [-2,1,-4,3]
        std::vector<float> in_data        = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int> pos_data         = {0};
        std::vector<float> cos_cache_data = {0.0f, 0.0f};
        std::vector<float> sin_cache_data = {1.0f, 1.0f};

        migraphx::parameter_map pp;
        pp["input"]     = migraphx::argument(in_shape, in_data.data());
        pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
        pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
        pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
        std::vector<float> expected = {-2.0f, 1.0f, -4.0f, 3.0f};
        EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
    }
}

TEST_CASE(rotary_embedding_verify_mixed_cos_sin_test)
{
    // cos=0, sin=1 applied to non-interleaved [1, 0, 0, 1] with D=4
    // signs = [-1,-1, 1,1], rotated = [0,1, 1,0]
    // output = [1,0,0,1]*0 + [-1,-1,1,1]*[0,1,1,0]*1 = [0,-1,1,0]
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 1, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {1, 2}};

    migraphx::module mm;
    auto input   = mm.add_parameter("input", in_shape);
    auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
    auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
    auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    std::vector<float> in_data        = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<int> pos_data         = {0};
    std::vector<float> cos_cache_data = {0.0f, 0.0f};
    std::vector<float> sin_cache_data = {1.0f, 1.0f};

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> expected = {0.0f, -1.0f, 1.0f, 0.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
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

TEST_CASE(rotary_embedding_4arg_offset_seq_gt1_test)
{
    // seq_len > 1 with non-zero start position: verifies gathernd uses pos_ids
    // rather than slicing [0..seq_len] from cache
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 2, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {4, 2}};

    migraphx::module mm;
    auto input   = mm.add_parameter("input", in_shape);
    auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
    auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
    auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    // cos_cache rows: [0.1,0.2], [0.3,0.4], [0.5,0.6], [0.7,0.8]
    // With pos_ids={2}, positions are 2 and 3 (start=2, seq_len=2).
    // sin=0 so output = input*cos, making gathered rows directly visible.
    std::vector<float> in_data(8, 1.0f);
    std::vector<int> pos_data         = {2};
    std::vector<float> cos_cache_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> sin_cache_data(8, 0.0f);

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Position 2: cos doubled=[0.5,0.6,0.5,0.6]; Position 3: cos doubled=[0.7,0.8,0.7,0.8]
    std::vector<float> expected = {0.5f, 0.6f, 0.5f, 0.6f, 0.7f, 0.8f, 0.7f, 0.8f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
}

TEST_CASE(rotary_embedding_4arg_per_batch_offset_test)
{
    // Different start positions per batch: batch 0 starts at 0, batch 1 starts at 3
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 1, 2, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {2}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {6, 2}};

    migraphx::module mm;
    auto input   = mm.add_parameter("input", in_shape);
    auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
    auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
    auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    // cos_cache rows: [1,0], [0,1], [-1,0], [0,-1], [0.5,0.5], [-0.5,-0.5]
    // Batch 0 at start=0 gathers rows 0,1; Batch 1 at start=3 gathers rows 3,4
    std::vector<float> in_data(16, 1.0f);
    std::vector<int> pos_data         = {0, 3};
    std::vector<float> cos_cache_data = {1, 0, 0, 1, -1, 0, 0, -1, 0.5f, 0.5f, -0.5f, -0.5f};
    std::vector<float> sin_cache_data(12, 0.0f);

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Batch 0: pos 0→[1,0,1,0], pos 1→[0,1,0,1]
    // Batch 1: pos 3→[0,-1,0,-1], pos 4→[0.5,0.5,0.5,0.5]
    std::vector<float> expected = {1, 0, 1, 0, 0, 1, 0, 1, 0, -1, 0, -1, 0.5f, 0.5f, 0.5f, 0.5f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
}

TEST_CASE(rotary_embedding_4arg_explicit_2d_positions_test)
{
    // Explicit per-token position IDs: [batch=2, seq=2]
    migraphx::shape in_shape{migraphx::shape::float_type, {2, 1, 2, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {2, 2}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {4, 2}};

    migraphx::module mm;
    auto input   = mm.add_parameter("input", in_shape);
    auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
    auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
    auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    // cos_cache rows: [1,0], [0,1], [-1,0], [0,-1]
    // Batch 0 positions: 3, 1; Batch 1 positions: 0, 2
    std::vector<float> in_data(16, 1.0f);
    std::vector<int> pos_data         = {3, 1, 0, 2};
    std::vector<float> cos_cache_data = {1, 0, 0, 1, -1, 0, 0, -1};
    std::vector<float> sin_cache_data(8, 0.0f);

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Batch 0: pos 3→[0,-1,0,-1], pos 1→[0,1,0,1]
    // Batch 1: pos 0→[1,0,1,0], pos 2→[-1,0,-1,0]
    std::vector<float> expected = {0, -1, 0, -1, 0, 1, 0, 1, 1, 0, 1, 0, -1, 0, -1, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
}

TEST_CASE(rotary_embedding_4arg_offset_nontrivial_cos_sin_test)
{
    // Full rotary computation with non-zero offset and non-trivial cos/sin
    migraphx::shape in_shape{migraphx::shape::float_type, {1, 1, 2, 4}};
    migraphx::shape pos_shape{migraphx::shape::int32_type, {1}};
    migraphx::shape cache_shape{migraphx::shape::float_type, {4, 2}};

    migraphx::module mm;
    auto input   = mm.add_parameter("input", in_shape);
    auto pos_ids = mm.add_parameter("pos_ids", pos_shape);
    auto cos_c   = mm.add_parameter("cos_cache", cache_shape);
    auto sin_c   = mm.add_parameter("sin_cache", cache_shape);
    migraphx::op::builder::add(
        "rotary_embedding", mm, {input, pos_ids, cos_c, sin_c}, {{"interleaved", false}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    // cos_cache rows: [1,1], [0.5,0.5], [1,1], [0,0]
    // sin_cache rows: [0,0], [0.5,0.5], [0,0], [1,1]
    // pos_ids={1} → positions 1 and 2
    std::vector<float> in_data        = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> pos_data         = {1};
    std::vector<float> cos_cache_data = {1, 1, 0.5f, 0.5f, 1, 1, 0, 0};
    std::vector<float> sin_cache_data = {0, 0, 0.5f, 0.5f, 0, 0, 1, 1};

    migraphx::parameter_map pp;
    pp["input"]     = migraphx::argument(in_shape, in_data.data());
    pp["pos_ids"]   = migraphx::argument(pos_shape, pos_data.data());
    pp["cos_cache"] = migraphx::argument(cache_shape, cos_cache_data.data());
    pp["sin_cache"] = migraphx::argument(cache_shape, sin_cache_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Position 1: cos=0.5, sin=0.5; input=[1,2,3,4]
    //   signs=[-1,-1,1,1], rotated=[3,4,1,2]
    //   = [1*0.5+(-1)*3*0.5, 2*0.5+(-1)*4*0.5, 3*0.5+1*1*0.5, 4*0.5+1*2*0.5]
    //   = [-1.0, -1.0, 2.0, 3.0]
    // Position 2: cos=1, sin=0; input=[5,6,7,8] → identity = [5,6,7,8]
    std::vector<float> expected = {-1.0f, -1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected));
}
