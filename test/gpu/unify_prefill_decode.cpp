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
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>

namespace {

migraphx::onnx_options unify_options()
{
    migraphx::onnx_options options;
    options.unify_prefill_decode          = true;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, 4};
    return options;
}

migraphx::parameter_map make_parameters(const migraphx::program& p,
                                        const std::vector<std::size_t>& lens,
                                        std::vector<float> data)
{
    migraphx::parameter_map params;
    const std::unordered_map<migraphx::sym::expr, std::size_t> bindings = {
        {migraphx::sym::var("sequence_length"), lens.at(1)}};
    const auto parameter_shapes = p.get_parameter_shapes();
    std::transform(parameter_shapes.begin(),
                   parameter_shapes.end(),
                   std::inserter(params, params.end()),
                   [&](const auto& parameter) {
                       auto s = parameter.second;
                       if(s.dynamic())
                           s = s.to_static(bindings);
                       return std::make_pair(parameter.first, migraphx::fill_argument(s, 0));
                   });
    migraphx::argument input{migraphx::shape{migraphx::shape::float_type, lens}};
    input.visit([&](auto output) { std::copy(data.begin(), data.end(), output.begin()); });
    params["x"] = std::move(input);
    return params;
}

std::vector<float> to_vector(const migraphx::argument& arg)
{
    std::vector<float> result;
    arg.visit([&](auto output) { result.assign(output.begin(), output.end()); });
    return result;
}

} // namespace

TEST_CASE(unify_prefill_decode_gpu)
{
    auto source = read_onnx("unify_prefill_decode_test.onnx", unify_options());
    auto ref    = source;
    ref.compile(migraphx::make_target("ref"));
    auto gpu = migraphx::make_target("gpu");
    source.compile(gpu);

    auto compare = [&](const std::vector<std::size_t>& lens, std::vector<float> data) {
        auto ref_params  = make_parameters(ref, lens, data);
        auto host_params = make_parameters(source, lens, std::move(data));
        migraphx::parameter_map gpu_params;
        std::transform(host_params.begin(),
                       host_params.end(),
                       std::inserter(gpu_params, gpu_params.end()),
                       [&](const auto& parameter) {
                           return std::make_pair(parameter.first, gpu.copy_to(parameter.second));
                       });
        auto gpu_result = gpu.copy_from(source.eval(gpu_params).back());
        auto ref_result = ref.eval(ref_params).back();
        EXPECT(migraphx::verify::verify_rms_range(to_vector(gpu_result), to_vector(ref_result)));
    };

    compare({1, 1, 2}, {1.0f, 2.0f});
    compare({1, 4, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
