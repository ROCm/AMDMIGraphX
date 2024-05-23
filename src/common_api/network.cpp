#include "MgxInfer.hpp"

#include <iostream>

using namespace mgxinfer1;

class Logger : public mgxinfer1::ILogger
{
    public:
    void log(Severity, mgxinfer1::AsciiChar const*) noexcept override {}
};

int main()
{
    auto logger = std::make_unique<Logger>();

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(*logger));
    if(!builder)
    {
        throw 1;
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if(!config)
    {
        throw 2;
    }

    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
    if(!network)
    {
        throw 3;
    }

    auto* x            = network->addInput("x", DataType::kFLOAT, Dims{2, {2, 3}});
    auto* reduce_layer = network->addReduce(*x, ReduceOperation::kSUM, 0x1, false);
    auto* reduce       = reduce_layer->getOutput(0);
    auto* abs          = network->addUnary(*reduce, UnaryOperation::kABS)->getOutput(0);
    auto* sin          = network->addUnary(*reduce, UnaryOperation::kSIN)->getOutput(0);
    // auto add = network->addElementWise(*abs, *sin, ElementWiseOperation::kSUM)->getOutput(0);
    // network->markOutput(*add);

    auto print_dims = [](const Dims dims) {
        std::cout << "[";
        for(auto i = 0; i < dims.nbDims; ++i)
        {
            std::cout << dims.d[i] << (i == dims.nbDims - 1 ? " " : ", ");
        }
        std::cout << "]\n";
    };

    // std::cout << "Output dims: ";
    // print_dims(add->getDimensions());

    std::cout << "Abs dims: ";
    print_dims(abs->getDimensions());
    std::cout << "Sin dims: ";
    print_dims(sin->getDimensions());

    reduce_layer->setReduceAxes(0x2);
    
    std::cout << "Abs dims: ";
    print_dims(abs->getDimensions());
    std::cout << "Sin dims: ";
    print_dims(sin->getDimensions());
    // std::cout << "Output dims: ";
    // print_dims(add->getDimensions());

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};

    return 0;
}
