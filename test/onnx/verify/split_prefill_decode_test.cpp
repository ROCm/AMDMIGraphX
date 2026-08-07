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

#include <onnx_test.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

namespace {

const auto ref_target = migraphx::make_target("ref");

migraphx::onnx_options split_options(std::size_t max_sequence_length)
{
    migraphx::onnx_options options;
    options.split_prefill_decode          = true;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, max_sequence_length};
    return options;
}

std::vector<float>
run(migraphx::program& p, const std::vector<std::size_t>& lens, std::vector<float> data)
{
    migraphx::parameter_map params;
    params["x"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, lens}, data.data()};

    std::vector<float> result;
    p.eval(params).back().visit([&](auto output) { result.assign(output.begin(), output.end()); });
    return result;
}

std::vector<float> to_vector(const migraphx::argument& arg)
{
    std::vector<float> result;
    arg.visit([&](auto output) { result.assign(output.begin(), output.end()); });
    return result;
}

bool verify_outputs(const std::vector<migraphx::argument>& results,
                    const std::vector<migraphx::argument>& expected)
{
    return results.size() == expected.size() and
           std::equal(results.begin(),
                      results.end(),
                      expected.begin(),
                      [](const auto& result, const auto& gold) {
                          return migraphx::verify::verify_rms_range(to_vector(result),
                                                                    to_vector(gold));
                      });
}

migraphx::parameter_map make_gqa_parameters(std::size_t sequence_length)
{
    migraphx::parameter_map result;
    result["qkv"] =
        migraphx::fill_argument({migraphx::shape::half_type, {1, sequence_length, 96}}, 0.5);
    result["key"]   = migraphx::fill_argument({migraphx::shape::float_type, {1}}, 0);
    result["value"] = migraphx::fill_argument({migraphx::shape::float_type, {1}}, 0);
    result["past_key_values_key"] =
        migraphx::fill_argument({migraphx::shape::half_type, {1, 2, 10, 16}}, 1);
    result["past_key_values_value"] =
        migraphx::fill_argument({migraphx::shape::half_type, {1, 2, 10, 16}}, 1);
    result["seqlens_k"] =
        migraphx::literal{{migraphx::shape::int32_type, {1, 1}}, {8}}.get_argument();
    return result;
}

} // namespace

// One compiled program serves both a single decode token and a full prefill sequence; the
// select_module picks the specialization from the shape of the input.
TEST_CASE(split_prefill_decode_test)
{
    auto p = read_onnx("split_prefill_decode_test.onnx", split_options(4));
    p.compile(ref_target);

    EXPECT(migraphx::verify::verify_rms_range(run(p, {1, 1, 2}, {1.0f, 2.0f}),
                                              std::vector<float>{2.0f, 3.0f}));
    EXPECT(migraphx::verify::verify_rms_range(
        run(p, {1, 4, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}),
        std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));
}

TEST_CASE(split_prefill_decode_intermediate_length_rejected_test)
{
    auto p = read_onnx("split_prefill_decode_test.onnx", split_options(4));
    p.compile(ref_target);

    EXPECT(test::throws([&] { run(p, {1, 2, 2}, std::vector<float>(4)); }));
}

TEST_CASE(split_prefill_decode_save_load_test)
{
    auto p = read_onnx("split_prefill_decode_test.onnx", split_options(4));
    p.compile(ref_target);
    p = migraphx::load_buffer(migraphx::save_buffer(p));

    EXPECT(migraphx::verify::verify_rms_range(run(p, {1, 1, 2}, {1.0f, 2.0f}),
                                              std::vector<float>{2.0f, 3.0f}));
    EXPECT(migraphx::verify::verify_rms_range(
        run(p, {1, 4, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}),
        std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));
}

TEST_CASE(split_prefill_decode_group_query_attention_test)
{
    auto split = read_onnx("group_query_attention_symbolic_test.onnx", split_options(8));
    split.compile(ref_target);

    auto verify_specialization = [&](std::size_t sequence_length) {
        migraphx::onnx_options fixed_options;
        fixed_options.use_symbolic_shapes           = true;
        fixed_options.dim_params["sequence_length"] = {sequence_length, sequence_length};
        auto fixed = read_onnx("group_query_attention_symbolic_test.onnx", fixed_options);
        fixed.compile(ref_target);

        auto params = make_gqa_parameters(sequence_length);
        EXPECT(verify_outputs(split.eval(params), fixed.eval(params)));
    };

    verify_specialization(1);
    verify_specialization(8);
}

TEST_CASE(split_prefill_decode_multi_io_test)
{
    auto options                          = split_options(4);
    options.dim_params["other_dimension"] = {2, 3};
    auto p = read_onnx("split_prefill_decode_multi_io_test.onnx", options);
    p.compile(ref_target);

    auto run_phase = [&](std::size_t sequence_length) {
        migraphx::parameter_map params;
        params["x"] =
            migraphx::fill_argument({migraphx::shape::float_type, {1, sequence_length, 2}}, 2);
        params["y"] =
            migraphx::fill_argument({migraphx::shape::float_type, {1, sequence_length, 2}}, 3);
        params["z"] = migraphx::fill_argument({migraphx::shape::float_type, {3, 2}}, 7);

        auto outputs = p.eval(params);
        EXPECT(outputs.size() == 2);
        EXPECT(migraphx::verify::verify_rms_range(to_vector(outputs.at(0)),
                                                  std::vector<float>(sequence_length * 2, 5)));
        EXPECT(
            migraphx::verify::verify_rms_range(to_vector(outputs.at(1)), std::vector<float>(6, 7)));
    };

    run_phase(1);
    run_phase(4);
}
