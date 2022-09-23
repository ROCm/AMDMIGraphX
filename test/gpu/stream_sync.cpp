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

#include <iostream>
#include <vector>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/target.hpp>
#include "test.hpp"

constexpr uint32_t STREAM_SYNC_TEST_VAL = 1337;

// NOLINTNEXTLINE
const std::string compare_numbers = R"__migraphx__(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void compare(float* data) 
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (data[i] != 1337) 
    {
        //abort();
    }
}
    
}

int main() {}

)__migraphx__";

migraphx::src_file make_src_file(const std::string& name, const std::string& content)
{
    return {name, std::make_pair(content.data(), content.data() + content.size())};
}

TEST_CASE(test_stream_sync_compare_kernel)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("check_stuff.cpp", compare_numbers)}, "", migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::gpu::kernel k1{binaries.front(), "compare"};

    auto input =
        migraphx::fill_argument({migraphx::shape::float_type, {128}}, STREAM_SYNC_TEST_VAL);

    auto ginput = migraphx::gpu::to_gpu(input);

    hipStream_t stream;
    auto status = hipStreamCreate(&stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to get stream");
    }

    k1.launch(stream, input.get_shape().elements(), 1024)(ginput.cast<std::float_t>());

    status = hipStreamDestroy(stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to cleanup stream");
    }

    auto output = migraphx::gpu::from_gpu(ginput);
    EXPECT(output == input);
}

TEST_CASE(test_stream_sync_different_stream)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("check_stuff.cpp", compare_numbers)}, "", migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::gpu::kernel k1{binaries.front(), "compare"};

    // Setup empty GPU memory buffer
    migraphx::shape io_shape{migraphx::shape::float_type, {128, 128}};
    auto input  = migraphx::fill_argument(io_shape, 0);
    auto ginput = migraphx::gpu::to_gpu(input);

    auto output  = migraphx::fill_argument(io_shape, 0);
    auto goutput = migraphx::gpu::to_gpu(output);

    hipStream_t stream;
    auto status = hipStreamCreate(&stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to get stream");
    }

    hipStream_t kernel_stream;
    status = hipStreamCreate(&kernel_stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to get stream2");
    }

    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto x        = mm->add_parameter("x", io_shape);
    auto y        = mm->add_literal({io_shape, {STREAM_SYNC_TEST_VAL + 100}});
    auto test_val = mm->add_literal({io_shape, {STREAM_SYNC_TEST_VAL}});

    auto mult_out = mm->add_instruction(migraphx::make_op("mul"), x, y);
    auto add_out  = mm->add_instruction(migraphx::make_op("add"), mult_out, test_val);
    mm->insert_parameter(add_out, "output", io_shape);

    migraphx::compile_options options;
    p.compile(migraphx::gpu::target{}, options);

    // Run network and then verify with kernel
    auto args = p.eval({{"x", ginput}, {"output", goutput}}, {stream, true});
    k1.launch(kernel_stream, input.get_shape().elements(), 1024)(goutput.cast<std::float_t>());

    output = migraphx::gpu::from_gpu(goutput);
    EXPECT(output != input);

    status = hipStreamDestroy(stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to cleanup stream");
    }

    status = hipStreamDestroy(kernel_stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to cleanup kernel stream");
    }
}

TEST_CASE(test_stream_sync_same_stream)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("check_stuff.cpp", compare_numbers)}, "", migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::gpu::kernel k1{binaries.front(), "compare"};

    // Setup empty GPU memory buffer
    migraphx::shape io_shape{migraphx::shape::float_type, {128, 128}};
    auto input  = migraphx::fill_argument(io_shape, 0);
    auto ginput = migraphx::gpu::to_gpu(input);

    auto output  = migraphx::fill_argument(io_shape, 0);
    auto goutput = migraphx::gpu::to_gpu(output);

    hipStream_t stream;
    auto status = hipStreamCreate(&stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to get stream");
    }

    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto x        = mm->add_parameter("x", io_shape);
    auto y        = mm->add_literal({io_shape, {STREAM_SYNC_TEST_VAL + 100}});
    auto test_val = mm->add_literal({io_shape, {STREAM_SYNC_TEST_VAL}});

    auto mult_out = mm->add_instruction(migraphx::make_op("mul"), x, y);
    auto add_out  = mm->add_instruction(migraphx::make_op("add"), mult_out, test_val);
    mm->insert_parameter(add_out, "output", io_shape);

    migraphx::compile_options options;
    p.compile(migraphx::gpu::target{}, options);

    // Run network and then verify with kernel
    auto args = p.eval({{"x", ginput}, {"output", goutput}}, {stream, true});
    k1.launch(stream, input.get_shape().elements(), 1024)(goutput.cast<std::float_t>());

    output = migraphx::gpu::from_gpu(goutput);
    EXPECT(output != input);

    status = hipStreamDestroy(stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to cleanup stream");
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
