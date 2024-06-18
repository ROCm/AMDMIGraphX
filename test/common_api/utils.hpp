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
#include <memory>
#include <MgxInfer.hpp>

class Logger : public mgxinfer1::ILogger
{
    public:
    void log(Severity, mgxinfer1::AsciiChar const*) noexcept override {}
};

struct layer_test_harness
{
    std::unique_ptr<mgxinfer1::ILogger> logger;
    std::unique_ptr<mgxinfer1::IBuilder> builder;
    std::unique_ptr<mgxinfer1::IBuilderConfig> builder_config;
    std::unique_ptr<mgxinfer1::INetworkDefinition> network;
    std::unique_ptr<mgxinfer1::IRuntime> runtime;
    std::unique_ptr<mgxinfer1::ICudaEngine> engine;
    std::unique_ptr<mgxinfer1::IExecutionContext> context;

    layer_test_harness()
    {
        logger  = std::make_unique<Logger>();
        builder = std::unique_ptr<mgxinfer1::IBuilder>(mgxinfer1::createInferBuilder(*logger));
        builder_config = std::unique_ptr<mgxinfer1::IBuilderConfig>(builder->createBuilderConfig());
        network = std::unique_ptr<mgxinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    }

    void make_infer()
    {
        auto plan = std::unique_ptr<mgxinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *builder_config));
        runtime = std::unique_ptr<mgxinfer1::IRuntime>(mgxinfer1::createInferRuntime(*logger));
        engine  = std::unique_ptr<mgxinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()));
        context = std::unique_ptr<mgxinfer1::IExecutionContext>(engine->createExecutionContext());
    }
};