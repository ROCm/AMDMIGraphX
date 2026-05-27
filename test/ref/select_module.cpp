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
#include <algorithm>
#include <cmath>

TEST_CASE(select_module_add_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6.0f}});

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto broadcast_lit =
            submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
        auto add_ins = submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
        submod->add_return({add_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2.0f, 14.0f, 5.0f, 10.0f, 5.0f, 14.0f, 14.0f, 2.0f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(select_module_reduce_test0)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-5, 12, 7, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(select_module_reduce_test1)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4, -4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {4, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-5, 12, 7, 4, -5, 12, 7, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(select_module_not_found_error)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8,  8,  -4, -4, 8,
                                  -1, 4, -1, 8, 8,  -4, -1, 8,  8,  -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {5, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

// Helper for the bucket-dispatch tests below. Builds a fresh program with
// `bucket_sizes` static submodules each computing `data + 6` and a
// select_module dispatch in the main module over the dynamic input shape.
static migraphx::program make_select_module_add_program(
    const std::vector<std::size_t>& bucket_sizes, std::size_t dyn_min, std::size_t dyn_max)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6.0f}});

    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto broadcast_lit =
            submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
        auto add_ins = submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
        submod->add_return({add_ins});
        return submod;
    };

    std::vector<migraphx::module_ref> submods;
    submods.reserve(bucket_sizes.size());
    std::transform(bucket_sizes.begin(),
                   bucket_sizes.end(),
                   std::back_inserter(submods),
                   [&](auto n) { return create_submodule(n, "bucket_" + std::to_string(n)); });

    migraphx::shape s{migraphx::shape::float_type, {{dyn_min, dyn_max}, {4, 4}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(
        migraphx::shape{migraphx::shape::float_type, {{dyn_min, dyn_max}, {4, 4}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        submods);
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    return p;
}

// Exact-shape match still works when bucket dispatch is enabled (the bucket
// path only kicks in when exact match fails).
TEST_CASE(select_module_bucket_dispatch_exact_match)
{
    // Buckets: 1, 10, 50, 100. Run with N=10 (exact match for bucket_10).
    auto p = make_select_module_add_program({1, 10, 50, 100}, 1, 100);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(10 * 4, 2.0f); // every element = 2
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {10, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(results_vector.size() == std::size_t{10 * 4});
    // Output should be input + 6 = 8 everywhere.
    EXPECT(std::all_of(
        // NOLINTNEXTLINE(clang-diagnostic-float-equal)
        results_vector.begin(),
        results_vector.end(),
        [](float v) { return std::fabs(v - 8.0f) < 1e-6f; }));
}

// Input N=20 with buckets {1, 10, 50, 100} should dispatch to the smallest
// compatible bucket (50). The first 20*4 outputs must match the
// reference; the padded tail (positions 20*4..50*4) sees zero-padded input,
// so output is 0 + 6 = 6.
TEST_CASE(select_module_bucket_dispatch_round_up_with_pad)
{
    auto p = make_select_module_add_program({1, 10, 50, 100}, 1, 100);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(20 * 4, 2.0f);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {20, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // Submodule produces a bucket-sized output (50 * 4) because we are
    // dispatching to dim_50.
    EXPECT(results_vector.size() == std::size_t{50 * 4});
    // First 20*4 positions correspond to real input (2 + 6 = 8).
    for(std::size_t i = 0; i < 20 * 4; ++i)
        EXPECT(std::fabs(results_vector[i] - 8.0f) < 1e-6f);
    // Remaining positions are zero-padded input (0 + 6 = 6).
    for(std::size_t i = 20 * 4; i < 50 * 4; ++i)
        EXPECT(std::fabs(results_vector[i] - 6.0f) < 1e-6f);
}

// Input N=200 has no compatible bucket in {1, 10, 50, 100}; dispatch must
// throw and not corrupt memory.
TEST_CASE(select_module_bucket_dispatch_too_big_throws)
{
    auto p = make_select_module_add_program({1, 10, 50, 100}, 1, 1000);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(200 * 4, 2.0f);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {200, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

// Hot-loop regression: drive the same dispatch 100 times in a row at the
// same input shape and verify every call produces the same output. The
// dispatch-cache hot path (introduced alongside this PR) lives in
// select_module::compute() and skips name lookups / map rebuilds on
// repeated hits; this test catches any cache-aliasing or stale-state
// bug.
TEST_CASE(select_module_bucket_dispatch_cache_hot_loop)
{
    auto p = make_select_module_add_program({1, 10, 50, 100}, 1, 100);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(10 * 4, 2.0f); // exact-match bucket=10
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {10, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());

    std::vector<float> first_result;
    for(int i = 0; i < 100; ++i)
    {
        auto r = p.eval(params).back();
        std::vector<float> v;
        r.visit([&](auto out) { v.assign(out.begin(), out.end()); });
        if(i == 0)
            first_result = v;
        else
            EXPECT(v.size() == first_result.size());
        EXPECT(std::equal(v.begin(), v.end(), first_result.begin(), [](float a, float b) {
            return std::fabs(a - b) < 1e-6f;
        }));
    }
    // Same model run 100 times must produce identical output throughout.
    EXPECT(first_result.size() == std::size_t{10 * 4});
    EXPECT(
        // NOLINTNEXTLINE(clang-diagnostic-float-equal)
        std::all_of(first_result.begin(), first_result.end(), [](float x) {
            return std::fabs(x - 8.0f) < 1e-6f;
        }));
}

// Hot-loop regression across SHAPES: alternate between two different
// runtime input shapes so the cache has to invalidate and re-populate.
// This catches a cache that returns stale data after a shape change.
TEST_CASE(select_module_bucket_dispatch_cache_alternating_shapes)
{
    auto p = make_select_module_add_program({1, 10, 50, 100}, 1, 100);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_a(10 * 4, 2.0f); // bucket=10
    std::vector<float> data_b(50 * 4, 3.0f); // bucket=50

    migraphx::shape shape_a{migraphx::shape::float_type, {10, 4}};
    migraphx::shape shape_b{migraphx::shape::float_type, {50, 4}};

    for(int i = 0; i < 50; ++i)
    {
        migraphx::parameter_map p_a;

        migraphx::parameter_map p_b;
        p_a["data"] = migraphx::argument(shape_a, data_a.data());
        p_b["data"] = migraphx::argument(shape_b, data_b.data());

        auto r_a = p.eval(p_a).back();
        auto r_b = p.eval(p_b).back();
        std::vector<float> v_a;

        std::vector<float> v_b;
        r_a.visit([&](auto out) { v_a.assign(out.begin(), out.end()); });
        r_b.visit([&](auto out) { v_b.assign(out.begin(), out.end()); });

        EXPECT(v_a.size() == std::size_t{10 * 4});
        // NOLINTNEXTLINE(clang-diagnostic-float-equal)
        EXPECT(std::all_of(
            v_a.begin(), v_a.end(), [](float x) { return std::fabs(x - 8.0f) < 1e-6f; }));
        EXPECT(v_b.size() == std::size_t{50 * 4});
        // NOLINTNEXTLINE(clang-diagnostic-float-equal)
        EXPECT(std::all_of(
            v_b.begin(), v_b.end(), [](float x) { return std::fabs(x - 9.0f) < 1e-6f; }));
    }
}
