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
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/op/external_weight.hpp>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include <test.hpp>

namespace {

const migraphx::shape input_shape{migraphx::shape::float_type, {4, 8}};
const migraphx::shape weight_shape{migraphx::shape::float_type, {8, 8}};
const std::string weight_file     = "weights.bin";
constexpr std::size_t num_weights = 3;

using weight_builder = std::function<migraphx::instruction_ref(migraphx::module&, std::size_t)>;

std::vector<migraphx::literal> make_weight_set(unsigned long seed)
{
    std::vector<migraphx::literal> result;
    std::generate_n(std::back_inserter(result), num_weights, [&] {
        return migraphx::generate_literal(weight_shape, seed++);
    });
    return result;
}

// All weights are stored back to back in a single file, as ONNX external data is.
migraphx::tmp_dir write_weight_set(const std::vector<migraphx::literal>& weights)
{
    migraphx::tmp_dir dir{"external_weights"};
    std::vector<char> buffer;
    for(const auto& w : weights)
        buffer.insert(buffer.end(), w.data(), w.data() + w.get_shape().bytes());
    migraphx::write_buffer(dir.path / weight_file, buffer);
    return dir;
}

// relu(x.w0 + x.w1).w2: the two independent dots give the scheduler parallel branches
// whose scratch buffers memory coloring can reuse.
migraphx::program make_program(const weight_builder& add_weight)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", input_shape);
    auto w0  = add_weight(*mm, 0);
    auto w1  = add_weight(*mm, 1);
    auto w2  = add_weight(*mm, 2);
    auto a   = mm->add_instruction(migraphx::make_op("dot"), x, w0);
    auto b   = mm->add_instruction(migraphx::make_op("dot"), x, w1);
    auto sum = mm->add_instruction(migraphx::make_op("add"), a, b);
    auto r   = mm->add_instruction(migraphx::make_op("relu"), sum);
    auto out = mm->add_instruction(migraphx::make_op("dot"), r, w2);
    mm->add_return({out});
    return p;
}

migraphx::program make_template()
{
    return make_program([](migraphx::module& m, std::size_t i) {
        return m.add_instruction(migraphx::op::external_weight{
            weight_shape, weight_file, i * weight_shape.bytes(), weight_shape.bytes()});
    });
}

std::vector<float> run(migraphx::program& p, const migraphx::argument& x)
{
    auto result = p.eval({{"x", x}}).back();
    std::vector<float> values;
    result.visit([&](auto v) { values.assign(v.begin(), v.end()); });
    return values;
}

std::vector<float> run_reference(const std::vector<migraphx::literal>& weights,
                                 const migraphx::argument& x)
{
    auto p = make_program(
        [&](migraphx::module& m, std::size_t i) { return m.add_literal(weights.at(i)); });
    p.compile(migraphx::make_target("ref"));
    return run(p, x);
}

std::vector<std::string> instruction_names(const migraphx::program& p)
{
    const auto* mm = p.get_main_module();
    std::vector<std::string> names;
    std::transform(mm->begin(), mm->end(), std::back_inserter(names), [](const auto& ins) {
        return ins.name();
    });
    return names;
}

std::size_t count_ops(const migraphx::program& p, const std::string& name)
{
    auto names = instruction_names(p);
    return std::count(names.begin(), names.end(), name);
}

} // namespace

TEST_CASE(encode_compiled_template)
{
    migraphx::compile_options options;
    options.offload_copy = true;
    auto t               = migraphx::make_target("gpu");
    auto tmpl            = make_template();
    tmpl.compile(t, options);
    EXPECT(count_ops(tmpl, "external_weight") == num_weights);
    const auto original = tmpl;

    auto x = migraphx::generate_argument(input_shape, 7);
    std::vector<std::vector<float>> outputs;
    for(unsigned long seed : {1, 100})
    {
        auto weights  = make_weight_set(seed);
        auto dir      = write_weight_set(weights);
        auto expected = run_reference(weights, x);

        auto encoded = migraphx::replace_onnx_external_weights(tmpl, dir.path.string(), t);
        EXPECT(count_ops(encoded, "external_weight") == 0);

        // Each external_weight is lowered in place: the scheduled order is untouched.
        auto names = instruction_names(tmpl);
        std::replace(names.begin(),
                     names.end(),
                     std::string{"external_weight"},
                     std::string{"gpu::literal"});
        EXPECT(instruction_names(encoded) == names);

        auto reloaded = migraphx::load_buffer(migraphx::save_buffer(encoded));
        auto result   = run(reloaded, x);
        EXPECT(migraphx::verify::verify_rms_range(result, expected));

        encoded.finalize();
        EXPECT(migraphx::verify::verify_rms_range(run(encoded, x), expected));

        outputs.push_back(result);
    }

    EXPECT(outputs.front() != outputs.back());
    EXPECT(tmpl == original);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
