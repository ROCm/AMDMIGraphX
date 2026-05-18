// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/instruction.hpp>

#include "Layer_impl.hpp"

namespace nvinfer1
{

Layer_impl::Layer_impl() noexcept
    : mType{-1}, mProgram{nullptr}
{
    ILayer::mLayer = this;
}

Layer_impl::Layer_impl(LayerType type, const std::shared_ptr<migraphx::program>& program) noexcept
    : mType{type}, mProgram{program}
{
    ILayer::mLayer = this;
}

Layer_impl::~Layer_impl() 
{
    pass_warning("TODO! implement me!", false);
}

LayerType Layer_impl::getType() const noexcept 
{
    return mType;
}

void Layer_impl::setName(char const* name) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

char const* Layer_impl::getName() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

int32_t Layer_impl::getNbInputs() const noexcept 
{
    return mInputs.size();
}

ITensor* Layer_impl::getInput(int32_t index) const noexcept 
{
    return mInputs.at(index);
}

int32_t Layer_impl::getNbOutputs() const noexcept 
{
    return mOutputs.size();
}

ITensor* Layer_impl::getOutput(int32_t index) const noexcept 
{
    return mOutputs.at(index).get();
}

void Layer_impl::setInput(int32_t index, ITensor& tensor) noexcept 
{
    Tensor_impl* tensorImpl = dynamic_cast<Tensor_impl*>(&tensor);
    
    auto* old_input = mInputs.at(index);
    mInputs[index]  = tensorImpl;

    migraphx::instruction::replace_argument(
        mInstructions.front(), old_input->getInstruction(), tensorImpl->getInstruction());
}

void Layer_impl::setPrecision(DataType dataType) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

DataType Layer_impl::getPrecision() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return DataType::kFLOAT;
}

bool Layer_impl::precisionIsSet() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void Layer_impl::resetPrecision() noexcept 
{
    pass_warning("TODO! implement me!", true);
}

void Layer_impl::setOutputType(int32_t index, DataType dataType) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

DataType Layer_impl::getOutputType(int32_t index) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return DataType::kFLOAT;
}

bool Layer_impl::outputTypeIsSet(int32_t index) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void Layer_impl::resetOutputType(int32_t index) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

void Layer_impl::setMetadata(char const* docString) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

char const* Layer_impl::getMetadata() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}
    
std::vector<migraphx::instruction_ref> Layer_impl::getInputArguments() const noexcept
{
    std::vector<migraphx::instruction_ref> args{};
    for (const auto& input : mInputs)
    {
        args.push_back(input->getInstruction());
    }
    return args;
}

} // namespace nvinfer1
