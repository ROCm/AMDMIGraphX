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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <test.hpp>

#include <numeric>

static void run_lowering(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::dead_code_elimination{}});
}

TEST_CASE(two_input_reshape_stays_on_host)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape output_shape{migraphx::shape::float_type, {6, 4}};

    migraphx::module m1;
    {
        auto input  = m1.add_parameter("input", input_shape);
        auto output = m1.add_parameter("output", output_shape);
        auto result = m1.add_instruction(migraphx::make_op("reshape"), input, output);
        m1.add_return({result});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto input       = m2.add_parameter("input", input_shape);
        auto output      = m2.add_parameter("output", output_shape);
        auto host_input  = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), input);
        auto host_output = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), output);
        auto synced_input =
            m2.add_instruction(migraphx::make_op("hip::sync_stream"), host_input, host_output);
        auto host_result =
            m2.add_instruction(migraphx::make_op("reshape"), synced_input, host_output);
        auto gpu_result =
            m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), host_result, output);
        m2.add_return({gpu_result});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(two_input_reshape_runs_on_host)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape output_shape{migraphx::shape::float_type, {6, 4}};
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input  = mm->add_parameter("input", input_shape);
    auto output = mm->add_parameter("output", output_shape);
    mm->add_return({mm->add_instruction(migraphx::make_op("reshape"), input, output)});

    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::make_target("gpu"), options);

    std::vector<float> input_data(input_shape.elements());
    std::iota(input_data.begin(), input_data.end(), 0.0f);
    std::vector<float> output_data(output_shape.elements());
    auto result = p.eval({{"input", migraphx::argument{input_shape, input_data.data()}},
                          {"output", migraphx::argument{output_shape, output_data.data()}}})
                      .back();
    EXPECT(result.get_shape() == output_shape);
    EXPECT(result.to_vector<float>() == input_data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
