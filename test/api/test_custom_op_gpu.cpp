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
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

#define MIGRAPHX_HIP_ASSERT(x) (EXPECT(x == hipSuccess))
struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // sets first half size_bytes of the input 0, and rest of the half bytes are copied.
        int* h_output    = nullptr;
        auto* d_output   = reinterpret_cast<int*>(inputs[0].data());
        auto input_bytes = inputs[0].get_shape().bytes();
        auto* output_ptr = inputs[1].data();
        auto copy_bytes  = input_bytes / 2;
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipHostMalloc(&h_output, input_bytes));
        MIGRAPHX_HIP_ASSERT(hipMemcpyAsync(
            h_output, d_output, input_bytes, hipMemcpyDeviceToHost, ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        MIGRAPHX_HIP_ASSERT(hipMemset(h_output, 0, copy_bytes));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        MIGRAPHX_HIP_ASSERT(hipMemcpy(output_ptr, h_output, input_bytes, hipMemcpyHostToDevice));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        MIGRAPHX_HIP_ASSERT(hipHostFree(h_output));
        return inputs[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        return inputs.back();
    }
};

TEST_CASE(run_simple_custom_op)
{
    simple_custom_op simple_op;
    migraphx::register_experimental_custom_op(simple_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_int32_type, {4, 3}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto neg           = m.add_instruction(migraphx::operation("neg"), x);
    auto alloc         = m.add_allocation(s);
    auto custom_kernel = m.add_instruction(migraphx::operation("simple_custom_op"), {neg, alloc});
    auto relu          = m.add_instruction(migraphx::operation("relu"), custom_kernel);
    m.add_return({relu});
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<int> x_data(12, -3);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results    = p.eval(pp);
    auto result     = results[0];
    auto result_vec = result.as_vector<int>();
    std::vector<int> expected_result(12, 0);
    std::fill(expected_result.begin() + 6, expected_result.end(), 3);
    EXPECT(bool{result == migraphx::argument(s, expected_result.data())});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
