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

#include <migraphx/manage_ptr.hpp>
#include "test.hpp"

TEST_CASE(load_and_run)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options options;
    options.set_offload_copy();
    options.set_exhaustive_tune_flag();
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

using hip_ptr    = MIGRAPHX_MANAGE_PTR(void, hipFree);
using stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

stream_ptr get_stream()
{
    hipStream_t stream;
    auto err = hipStreamCreateWithFlags(&stream, 0);
    EXPECT(err == hipSuccess);
    return stream_ptr{stream};
}

hip_ptr get_hip_buffer(size_t size)
{
    void* ptr;
    auto err = hipMalloc(&ptr, size);
    EXPECT(err == hipSuccess);
    return hip_ptr{ptr};
}

TEST_CASE(dynamic_batch_load_and_run)
{
    migraphx::onnx_options o_options;
    migraphx::dynamic_dimensions dyn_dims = {{1, 4, {2, 4}}, {3, 3}, {4, 4}, {4, 4}};
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
    dyn_dims = {{2, 2}, {3, 3}, {3, 3}, {3, 3}};
    o_options.set_dyn_input_parameter_shape("1", dyn_dims);
    auto p = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", o_options);
    migraphx::compile_options c_options;
    c_options.set_split_single_dyn_dim();
    p.compile(migraphx::target("gpu"), c_options);
    auto out_shapes = p.get_output_shapes();
    CHECK(out_shapes.size() == 1);
    EXPECT(out_shapes[0].dynamic());

    std::vector<float> a = {
        2.71567607,  -0.9960829,  0.91671127,  0.28140706,  0.63235772,  0.08077253,  0.80927712,
        -0.59108931, -1.05421555, -2.76622486, -0.85044265, -0.52049929, 0.67726439,  -0.65290606,
        0.02345525,  -0.33579525, 0.38901961,  1.05473483,  -1.31188095, 1.8963089,   -0.07265259,
        0.947339,    0.41949373,  -0.70814759, 0.25892952,  1.07311416,  1.2571274,   -0.62318051,
        -0.19951548, -0.94232577, -0.29393643, 0.42292568,  -0.80230367, 1.40909171,  0.63617158,
        0.13900366,  1.09253144,  -0.15265895, 1.54781747,  0.72780299,  1.09189606,  -0.38068101,
        0.97057933,  -0.58958799, 1.56188643,  0.21474874,  0.58725154,  -1.27097559, -0.03024297,
        1.09437096,  -0.4897908,  0.34838957,  -1.31042492, -1.69069934, 0.86956722,  -0.40457946,
        0.46691212,  1.29273605,  0.26464137,  0.22073045,  -1.02178168, 0.22163901,  -1.84387338,
        0.75522131,  -0.45775682, -0.42241111, -1.50944722, 1.07256448,  -1.95876884, -0.28106022,
        0.3341668,   2.13129425,  -1.14728117, -1.06555498, -0.298444,   -0.88322699, -0.65866792,
        -2.06007552, 0.01374334,  0.45612028,  0.52715492,  1.01914406,  -1.72659791, 0.80650896,
        0.16860051,  2.24112225,  -0.78620857, 0.36566174,  -0.07020134, -0.47976932, -0.68230027,
        -0.94711417, -0.54506505, 1.66504931,  -0.71860826, 0.61132306};

    std::vector<float> c = {
        -0.14601797, -0.13000923, 0.06521662,  0.06178288,  -0.11083675, 0.10154136,  0.09990512,
        0.06030385,  -0.11374587, -0.17523311, -0.14344215, 0.17802463,  0.06300922,  -0.15325832,
        0.07066704,  0.05166031,  0.00615084,  -0.02606523, 0.08083995,  -0.17913306, 0.0624622,
        0.0735731,   -0.04198661, -0.0164391,  -0.06374192, 0.16569914,  0.10681538,  0.07370754,
        0.02802075,  0.00282027,  0.15104802,  -0.11084409, -0.00197773, 0.07924436,  0.03528272,
        0.04765259,  -0.15896152, 0.07917164,  0.12125669,  -0.1154705,  -0.11999125, 0.12749968,
        -0.06269585, 0.18658121,  -0.03944227, 0.0111798,   -0.17731084, 0.11789055,  -0.09982193,
        0.08142821,  0.0729029,   0.11303909,  0.12735154,  0.03885292};

    auto param_shapes = p.get_parameter_shapes();
    int batch_size    = 2;
    std::unordered_map<std::string, migraphx::argument> arg_map;

    arg_map["0"] = migraphx::argument(param_shapes["0"].to_static(batch_size), a.data());
    arg_map["1"] = migraphx::argument(param_shapes["1"].to_static(batch_size), c.data());

    migraphx::program_parameters pp;
    std::vector<hip_ptr> buffs;
    std::vector<migraphx::argument> args;

    // copy to GPU and create parameter map
    for(auto&& name : param_shapes.names())
    {
        if(arg_map.find(name) != arg_map.end())
        {
            args.push_back(arg_map.at(name));
        }
        else
        {
            migraphx::shape static_shape = param_shapes[name].to_static(batch_size);
            auto output_arg              = migraphx::argument(static_shape);
            args.push_back(output_arg);
        }
        buffs.push_back(get_hip_buffer(args.rbegin()->get_shape().bytes()));
        auto err = hipMemcpy(buffs.rbegin()->get(),
                             args.rbegin()->data(),
                             args.rbegin()->get_shape().bytes(),
                             hipMemcpyHostToDevice);
        EXPECT(err == hipSuccess);
        pp.add(name, migraphx::argument(args.rbegin()->get_shape(), buffs.rbegin()->get()));
    }

    auto output = p.eval(pp)[0];

    // copy output back to host
    auto host_arg = migraphx::argument(output.get_shape());
    auto err      = hipMemcpy(
        host_arg.data(), output.data(), output.get_shape().bytes(), hipMemcpyDeviceToHost);
    EXPECT(err == hipSuccess);

    std::vector<float> sol = {-0.20817225,
                              0.87965256,
                              0.14958936,
                              -1.24887264,
                              -0.06540672,
                              0.20778663,
                              0.40456355,
                              -0.99900877,
                              0.4917807,
                              0.1994698,
                              0.64205718,
                              0.37798831,
                              -0.25315839,
                              0.44276932,
                              -0.16138598,
                              0.79344082};
    EXPECT(host_arg.as_vector<float>() == sol);
}

TEST_CASE(load_and_run_async)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options options;
    options.set_offload_copy(false);
    p.compile(migraphx::target("gpu"), options);
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();

    stream_ptr stream = get_stream();

    std::vector<hip_ptr> buffs;
    std::vector<migraphx::argument> args;
    for(auto&& name : param_shapes.names())
    {
        args.push_back(migraphx::argument::generate(param_shapes[name]));
        buffs.push_back(get_hip_buffer(args.rbegin()->get_shape().bytes()));

        auto err = hipMemcpy(buffs.rbegin()->get(),
                             args.rbegin()->data(),
                             args.rbegin()->get_shape().bytes(),
                             hipMemcpyHostToDevice);
        EXPECT(err == hipSuccess);
        pp.add(name, migraphx::argument(args.rbegin()->get_shape(), buffs.rbegin()->get()));
    }

    auto outputs = p.run_async(pp, stream.get());
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
        std::vector<float> xd(xs.elements(), 1.0);
        pp.add("x", migraphx::argument(xs, xd.data()));
        auto ys = param_shapes["y"];
        std::vector<float> yd(ys.elements(), 2.0);
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
