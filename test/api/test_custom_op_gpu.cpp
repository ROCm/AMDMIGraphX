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
#include <migraphx/verify.hpp>
#include <migraphx/migraphx.hpp>
#include <stdexcept>
#include "test.hpp"

#define MIGRAPHX_HIP_ASSERT(x) (EXPECT(x == hipSuccess))
struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }

    virtual bool runs_on_offload_target() const override { return false; }

    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // This custom op simply sets first half size_bytes of the input 0, and rest of the half
        // bytes are copied. for this custom_op, it does its computation on the host. Therefore,
        // `runs_on_offload_target()` is set to false. MIGraphX would inject necessary buffer copies
        // to and from GPU to Host based on `runs_on_offload_targe()` flag for input buffers as well
        // as the output buffers
        auto* input_buffer_ptr  = inputs[0].data();
        auto* output_buffer_ptr = inputs[1].data();
        auto input_bytes        = inputs[0].get_shape().bytes();
        auto copy_bytes         = input_bytes / 2;
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipMemcpyAsync(output_buffer_ptr,
                                           input_buffer_ptr,
                                           input_bytes,
                                           hipMemcpyHostToHost,
                                           ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        MIGRAPHX_HIP_ASSERT(hipMemset(output_buffer_ptr, 0, copy_bytes));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        return inputs[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(!inputs[0].standard())
        {
            throw std::runtime_error("first arg must be standard shaped");
        }
        if(inputs.size() != 2)
        {
            throw std::runtime_error("number of inputs must be 2");
        }
        return inputs.back();
    }
};
struct transpose_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "transpose_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context, migraphx::shape out_shape, migraphx::arguments inputs) const override
    {
        return {out_shape, inputs[0].data()};
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 1)
        {
            throw std::runtime_error("transpose custom op must have only one input argument");
        };
        migraphx::shape input_s     = inputs[0];
        std::vector<size_t> dims    = input_s.lengths();
        std::vector<size_t> strides = input_s.strides();
        std::reverse(dims.begin(), dims.end());
        std::reverse(strides.begin(), strides.end());
        migraphx::shape output_shape{input_s.type(), dims, strides};
        return output_shape;
    }

    virtual bool runs_on_offload_target() const override { return true; }
    virtual std::ptrdiff_t output_alias(migraphx::shapes) const override { return 0; };
};

TEST_CASE(run_simple_custom_op)
{
    simple_custom_op simple_op;
    migraphx::register_experimental_custom_op(simple_op);
    transpose_custom_op transpose_op;
    migraphx::register_experimental_custom_op(transpose_op);

    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {4, 3}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto neg           = m.add_instruction(migraphx::operation("neg"), x);
    auto alloc         = m.add_allocation(s);
    auto alloc2        = m.add_allocation(migraphx::shape(migraphx_shape_float_type, {3, 4}));
    auto custom_kernel = m.add_instruction(migraphx::operation("simple_custom_op"), {neg, alloc});
    auto custom_transpose =
        m.add_instruction(migraphx::operation("transpose_custom_op"), {custom_kernel});
    auto relu     = m.add_instruction(migraphx::operation("relu"), custom_transpose);
    auto cont_ins = m.add_instruction(migraphx::operation("contiguous"), relu);
    auto custom_kernel2 =
        m.add_instruction(migraphx::operation("simple_custom_op"), {cont_ins, alloc2});
    auto neg2 = m.add_instruction(migraphx::operation("neg"), custom_kernel2);
    m.add_return({neg2});
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(12, -3);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    auto result_vec                    = result.as_vector<float>();
    std::vector<float> expected_result = {0, 0, 0, 0, 0, 0, -3, -3, 0, 0, -3, -3};
    EXPECT(bool{result == migraphx::argument(migraphx::shape(migraphx_shape_float_type, {3, 4}),
                                             expected_result.data())});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
