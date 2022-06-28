/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <numeric>
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_and_run)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(load_and_run_ctx)
{
    auto p = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto ctx = p.experimental_get_context();
    EXPECT(ctx.get_queue<hipStream_t>() != nullptr);
    p.eval(pp);
    ctx.finish();
}

TEST_CASE(if_pl_test)
{
    auto run_prog = [&](auto cond) {
        auto p             = migraphx::parse_onnx("if_pl_test.onnx");
        auto shapes_before = p.get_output_shapes();
        migraphx::compile_options options;
        options.set_offload_copy();
        p.compile(migraphx::target("gpu"), options);
        auto shapes_after = p.get_output_shapes();
        CHECK(shapes_before.size() == 1);
        CHECK(bool{shapes_before.front() == shapes_after.front()});

        migraphx::program_parameters pp;
        auto param_shapes = p.get_parameter_shapes();
        auto xs           = param_shapes["x"];
        std::vector<float> xd(xs.bytes() / sizeof(float), 1.0);
        pp.add("x", migraphx::argument(xs, xd.data()));
        auto ys = param_shapes["y"];
        std::vector<float> yd(ys.bytes() / sizeof(float), 2.0);
        pp.add("y", migraphx::argument(ys, yd.data()));
        char ccond = cond;
        pp.add("cond", migraphx::argument(param_shapes["cond"], &ccond));

        auto outputs = p.eval(pp);
        auto output  = outputs[0];
        return output.as_vector<float>();
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {2, 3, 4, 5, 6, 7};
        EXPECT(result_vector == gold);
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {1, 2, 3, 4, 5, 6};
        EXPECT(result_vector == gold);
    }
}

TEST_CASE(loop_test)
{
    auto run_prog = [&](int64_t max_iter_num) {
        migraphx::onnx_options parse_options;
        parse_options.set_default_loop_iterations(max_iter_num);
        auto p             = migraphx::parse_onnx("loop_default_test.onnx", parse_options);
        auto shapes_before = p.get_output_shapes();
        migraphx::compile_options options;
        options.set_offload_copy();
        p.compile(migraphx::target("gpu"), options);
        auto shapes_after = p.get_output_shapes();
        CHECK(shapes_before.size() == 2);
        CHECK(bool{shapes_before.front() == shapes_after.front()});

        migraphx::program_parameters pp;
        auto param_shapes     = p.get_parameter_shapes();
        auto aas              = param_shapes["a"];
        std::vector<float> xd = {1.0f};
        pp.add("a", migraphx::argument(aas, xd.data()));
        auto bbs              = param_shapes["b"];
        std::vector<float> yd = {2.0};
        pp.add("b", migraphx::argument(bbs, yd.data()));

        auto outputs = p.eval(pp);
        auto output  = outputs[0];
        std::vector<std::vector<float>> ret;
        ret.push_back(output.as_vector<float>());

        output = outputs[1];
        ret.push_back(output.as_vector<float>());

        return ret;
    };

    {
        auto result_vector       = run_prog(10);
        std::vector<float> gold0 = {2.0f};
        EXPECT(result_vector.at(0) == gold0);
        std::vector<float> gold1 = {-2, 4, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT(result_vector.at(1) == gold1);
    }

    {
        auto result_vector       = run_prog(15);
        std::vector<float> gold0 = {2.0f};
        EXPECT(result_vector.at(0) == gold0);
        std::vector<float> gold1 = {-2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT(result_vector.at(1) == gold1);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
