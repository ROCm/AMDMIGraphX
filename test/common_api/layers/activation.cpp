/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include "MgxInfer.hpp"
#include <hip/hip_runtime_api.h>
#include <migraphx/stringutils.hpp>
#include <utils.hpp>

#include <test.hpp>

TEST_CASE(activation_layer_relu_creation)
{
    layer_test_harness harness;
    auto& network = harness.network;

    auto x_dims = mgxinfer1::Dims{1, {3}};
    auto* x     = network->addInput("x", mgxinfer1::DataType::kFLOAT, x_dims);
    auto* relu  = network->addActivation(*x, mgxinfer1::ActivationType::kRELU);
    network->markOutput(*relu->getOutput(0));

    harness.make_infer();

    float *x_data_dev = nullptr, *output_data_dev = nullptr;
    hipMalloc(&x_data_dev, mgxinfer1::volume(x_dims) * sizeof(float));
    hipMalloc(&output_data_dev, mgxinfer1::volume(x_dims) * sizeof(float));
    hipStream_t stream;
    hipStreamCreate(&stream);

    std::vector<float> x_data{-1.f, 0.f, 1.f};
    hipMemcpyAsync(
        x_data_dev, x_data.data(), x_data.size() * sizeof(float), hipMemcpyHostToDevice, stream);

    harness.context->setTensorAddress("x", x_data_dev);
    harness.context->setTensorAddress("output", output_data_dev);
    harness.context->enqueueV3(stream);

    hipMemcpyAsync(x_data.data(),
                   output_data_dev,
                   x_data.size() * sizeof(float),
                   hipMemcpyDeviceToHost,
                   stream);
    hipStreamSynchronize(stream);
    std::vector<float> gold{0.f, 0.f, 1.f};
    EXPECT(migraphx::verify::verify_rms_range(x_data, gold));

    hipFree(x_data_dev);
    hipFree(output_data_dev);
    hipStreamDestroy(stream);
}
