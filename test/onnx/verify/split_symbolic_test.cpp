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
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/verify.hpp>
#include <algorithm>
#include <numeric>

namespace {

migraphx::onnx_options symbolic_options(std::size_t max_sequence_length)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes           = true;
    options.dim_params["sequence_length"] = {1, max_sequence_length};
    return options;
}

migraphx::onnx_options static_options(const std::unordered_map<std::string, std::size_t>& dims)
{
    migraphx::onnx_options options;
    std::transform(dims.begin(),
                   dims.end(),
                   std::inserter(options.dim_params, options.dim_params.end()),
                   [](const auto& dim) {
                       return std::make_pair(
                           dim.first, migraphx::shape::dynamic_dimension{dim.second, dim.second});
                   });
    return options;
}

// A prompt is processed all at once and then one token at a time, so those are the only two
// sequence lengths worth building a specialization for.
void split_sequence_length(migraphx::program& p, std::size_t max_sequence_length)
{
    migraphx::run_passes(
        p,
        {migraphx::split_single_dyn_dim{"sequence_length", {1, max_sequence_length}},
         migraphx::dead_code_elimination{}});
}

std::vector<migraphx::module_ref> find_specializations(migraphx::const_module_ref mod)
{
    auto select = std::find_if(
        mod->begin(), mod->end(), [](const auto& ins) { return ins.name() == "select_module"; });
    if(select == mod->end())
        return {};
    return select->module_inputs();
}

std::size_t count_literals(migraphx::const_module_ref mod)
{
    auto instructions = migraphx::iterator_for(*mod);
    return std::count_if(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "@literal";
    });
}

migraphx::argument ramp(const migraphx::shape& s, float start)
{
    migraphx::argument result{s};
    result.visit([&](auto view) { std::iota(view.begin(), view.end(), start); });
    return result;
}

// A specialization has to compute what the model computes when it is parsed for that sequence
// length in the first place. A symbolic program is not evaluated directly -- specializing it is
// what makes it runnable -- so the reference is the statically parsed model.
void expect_matches_static_parse(migraphx::program split,
                                 const std::string& model,
                                 const migraphx::onnx_options& static_options,
                                 const migraphx::parameter_map& params)
{
    auto reference = read_onnx(model, static_options);

    split.compile(migraphx::make_target("ref"));
    reference.compile(migraphx::make_target("ref"));

    auto split_results     = split.eval(params);
    auto reference_results = reference.eval(params);
    EXPECT(split_results.size() == reference_results.size());
    for(auto i : migraphx::range(split_results.size()))
    {
        EXPECT(split_results.at(i).get_shape() == reference_results.at(i).get_shape());
        EXPECT(migraphx::verify::verify_rms_range(split_results.at(i).to_vector<float>(),
                                                  reference_results.at(i).to_vector<float>()));
    }
}

} // namespace

TEST_CASE(split_symbolic_sequence_length)
{
    auto p = read_onnx("split_prefill_decode_test.onnx", symbolic_options(4));
    split_sequence_length(p, 4);

    auto* mm             = p.get_main_module();
    auto specializations = find_specializations(mm);
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 1, 2}});
    EXPECT(specializations.at(1)->get_parameter_shape("x") ==
           migraphx::shape{migraphx::shape::float_type, {1, 4, 2}});

    // The initializer stays in the main module and is captured by both specializations rather
    // than copied into each of them.
    EXPECT(count_literals(mm) == 1);
    EXPECT(count_literals(specializations.at(0)) == 0);
    EXPECT(count_literals(specializations.at(1)) == 0);

    // The main module keeps the symbolic shapes, so it still reports the exact output shape for
    // whichever sequence length shows up.
    EXPECT(mm->get_parameter_shape("x").symbolic());
    EXPECT(mm->get_output_shapes().at(0).symbolic());

    // a capture crosses a module boundary, so it has to survive a serialization round trip
    migraphx::program reloaded;
    reloaded.from_value(p.to_value());
    EXPECT(reloaded == p);

    for(std::size_t sequence_length : {std::size_t{1}, std::size_t{4}})
    {
        migraphx::shape xs{migraphx::shape::float_type, {1, sequence_length, 2}};
        expect_matches_static_parse(p,
                                    "split_prefill_decode_test.onnx",
                                    static_options({{"sequence_length", sequence_length}}),
                                    {{"x", ramp(xs, 1.0f)}});
    }
}

// Specializing the sequence length says nothing about a dimension that varies on its own: it has
// to stay symbolic in the specializations and in the program's output shapes.
TEST_CASE(split_symbolic_independent_dimension)
{
    auto options                          = symbolic_options(4);
    options.dim_params["other_dimension"] = {2, 3};
    auto p = read_onnx("split_prefill_decode_multi_io_test.onnx", options);
    split_sequence_length(p, 4);

    auto* mm             = p.get_main_module();
    auto specializations = find_specializations(mm);
    EXPECT(specializations.size() == 2);
    EXPECT(specializations.at(0)->get_parameter_shape("z").symbolic());
    EXPECT(specializations.at(0)->get_parameter_shape("z") ==
           specializations.at(1)->get_parameter_shape("z"));
    EXPECT(not specializations.at(0)->get_parameter_shape("x").dynamic());

    auto output_shapes = mm->get_output_shapes();
    EXPECT(output_shapes.size() == 2);
    EXPECT(output_shapes.at(1).symbolic());
    EXPECT(output_shapes.at(1) == mm->get_parameter_shape("z"));

    // The independent dimension still varies, so the specialization built for one sequence length
    // has to serve every size it takes.
    for(std::size_t sequence_length : {std::size_t{1}, std::size_t{4}})
    {
        for(std::size_t other : {std::size_t{2}, std::size_t{3}})
        {
            migraphx::shape xs{migraphx::shape::float_type, {1, sequence_length, 2}};
            migraphx::shape zs{migraphx::shape::float_type, {other, 2}};
            expect_matches_static_parse(
                p,
                "split_prefill_decode_multi_io_test.onnx",
                static_options({{"sequence_length", sequence_length}, {"other_dimension", other}}),
                {{"x", ramp(xs, 1.0f)}, {"y", ramp(xs, 10.0f)}, {"z", ramp(zs, 100.0f)}});
        }
    }
}
