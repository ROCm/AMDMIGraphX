// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "Helper.hpp"
#include "ConstantLayer_impl.hpp"
#include <memory>

namespace nvinfer1
{

ConstantLayer_impl::ConstantLayer_impl() noexcept
    : Layer_impl{LayerType::kCONSTANT, nullptr}, mDimensions{}, mWeights{DataType::kFLOAT, nullptr, 0}
{
    pass_warning("TODO! implement me!", false);
    IConstantLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IConstantLayer::mImpl = this;
}

ConstantLayer_impl::ConstantLayer_impl(Dims const& dimensions, Weights weights, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kCONSTANT, program}, mDimensions{dimensions}, mWeights{weights}
{
    IConstantLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IConstantLayer::mImpl = this;
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

void ConstantLayer_impl::build() noexcept
{
    auto* mm = getModule();
    migraphx::shape s = helper::dimsToShape(getWeights().type, getDimensions());
    auto buff = reinterpret_cast<const uint8_t*>(getWeights().values);
    mInstructions.push_back(mm->add_literal(s, buff));
  
    mOutputs[0]->setInstruction(mInstructions.back());
}

ConstantLayer_impl::~ConstantLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

void ConstantLayer_impl::setWeights(Weights weights) noexcept
{
    mWeights = weights;
}

Weights ConstantLayer_impl::getWeights() const noexcept
{
    return mWeights;
}

void ConstantLayer_impl::setDimensions(Dims const& dimensions) noexcept
{
    mDimensions = dimensions;
}

Dims ConstantLayer_impl::getDimensions() const noexcept
{
    return mDimensions;
}

} // namespace nvinfer1
