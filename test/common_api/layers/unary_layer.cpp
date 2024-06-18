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

#include <test.hpp>

class Logger : public mgxinfer1::ILogger
{
    public:
    void log(Severity, mgxinfer1::AsciiChar const*) noexcept override {}
};

TEST_CASE(unary_layer_creation)
{
    auto logger  = std::make_unique<Logger>();
    auto builder = std::unique_ptr<mgxinfer1::IBuilder>(mgxinfer1::createInferBuilder(*logger));
    auto network = std::unique_ptr<mgxinfer1::INetworkDefinition>(builder->createNetworkV2(0));

    auto* x   = network->addInput("x", mgxinfer1::DataType::kFLOAT, mgxinfer1::Dims{2, {2, 3}});
    auto* sin = network->addUnary(*x, mgxinfer1::UnaryOperation::kSIN);
    network->markOutput(*sin->getOutput(0));

    auto config = std::unique_ptr<mgxinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto plan =
        std::unique_ptr<mgxinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    auto runtime = std::unique_ptr<mgxinfer1::IRuntime>(mgxinfer1::createInferRuntime(*logger));
    auto engine  = std::shared_ptr<mgxinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    auto context = std::unique_ptr<mgxinfer1::IExecutionContext>(engine->createExecutionContext());

    float *x_data_dev = nullptr, *output_data_dev = nullptr;
    hipMalloc(&x_data_dev, 2 * 3 * sizeof(float));
    hipMalloc(&output_data_dev, 2 * 3 * sizeof(float));
    hipStream_t stream;
    hipStreamCreate(&stream);

    std::vector<float> x_data{
        0.38335552, 0.99845273, 0.28325482, 0.57934947, 0.95479255, 0.49634385};
    // TODO replace with memcpy async
    hipMemcpyAsync(
        x_data_dev, x_data.data(), x_data.size() * sizeof(float), hipMemcpyHostToDevice, stream);

    context->setTensorAddress("x", x_data_dev);
    context->setTensorAddress("output", output_data_dev);
    context->enqueueV3(stream);

    hipMemcpyAsync(x_data.data(),
                   output_data_dev,
                   x_data.size() * sizeof(float),
                   hipMemcpyDeviceToHost,
                   stream);
    hipStreamSynchronize(stream);
    std::vector<float> gold{0.37403453, 0.84063398, 0.27948225, 0.54747968, 0.8161939, 0.47621377};
    EXPECT(migraphx::verify::verify_rms_range(x_data, gold));

    hipFree(x_data_dev);
    hipFree(output_data_dev);
    hipStreamDestroy(stream);
}

TEST_CASE(unary_layer_modification)
{
    auto logger  = std::make_unique<Logger>();
    auto builder = std::unique_ptr<mgxinfer1::IBuilder>(mgxinfer1::createInferBuilder(*logger));
    auto network = std::unique_ptr<mgxinfer1::INetworkDefinition>(builder->createNetworkV2(0));

    auto* x   = network->addInput("x", mgxinfer1::DataType::kFLOAT, mgxinfer1::Dims{2, {2, 3}});
    auto* sin = network->addUnary(*x, mgxinfer1::UnaryOperation::kSIN);
    network->markOutput(*sin->getOutput(0));

    sin->setOperation(mgxinfer1::UnaryOperation::kCOS);

    auto config = std::unique_ptr<mgxinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto plan =
        std::unique_ptr<mgxinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    auto runtime = std::unique_ptr<mgxinfer1::IRuntime>(mgxinfer1::createInferRuntime(*logger));
    auto engine  = std::shared_ptr<mgxinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    auto context = std::unique_ptr<mgxinfer1::IExecutionContext>(engine->createExecutionContext());

    float *x_data_dev = nullptr, *output_data_dev = nullptr;
    hipMalloc(&x_data_dev, 2 * 3 * sizeof(float));
    hipMalloc(&output_data_dev, 2 * 3 * sizeof(float));
    hipStream_t stream;
    hipStreamCreate(&stream);

    std::vector<float> x_data{
        0.38335552, 0.99845273, 0.28325482, 0.57934947, 0.95479255, 0.49634385};
    hipMemcpyAsync(
        x_data_dev, x_data.data(), x_data.size() * sizeof(float), hipMemcpyHostToDevice, stream);

    context->setTensorAddress("x", x_data_dev);
    context->setTensorAddress("output", output_data_dev);
    context->enqueueV3(stream);

    hipMemcpyAsync(x_data.data(),
                   output_data_dev,
                   x_data.size() * sizeof(float),
                   hipMemcpyDeviceToHost,
                   stream);
    hipStreamSynchronize(stream);
    std::cout << migraphx::to_string_range(x_data) << std::endl;
    std::vector<float> gold{0.92741478, 0.54160364, 0.96015086, 0.83681898, 0.57777809, 0.87932954};
    EXPECT(migraphx::verify::verify_rms_range(x_data, gold));

    hipFree(x_data_dev);
    hipFree(output_data_dev);
    hipStreamDestroy(stream);
}
