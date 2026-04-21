// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "Helper.hpp"
#include "ConstantLayer_impl.hpp"
#include <memory>

namespace nvinfer1
{

ConstantLayer_impl::ConstantLayer_impl() noexcept
{
    pass_warning("TODO! implement me!", false);
    IConstantLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));;
    IConstantLayer::mImpl = this;
}

ConstantLayer_impl::ConstantLayer_impl(Dims const& dimensions, Weights weights, const std::shared_ptr<migraphx::program>& program) noexcept
{
    pass_warning("TODO! implement me!", false);
    IConstantLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));;
    IConstantLayer::mImpl = this;

    auto* mm = program->get_main_module();
    mInstructions.push_back( 
        mm->add_literal(migraphx::shape{helper::fromDataType(weights.type), helper::dimsToVec(dimensions)},
                            reinterpret_cast<const uint8_t*>(weights.values)));
    
    mOutputs.emplace_back(std::make_unique<Tensor_impl>(mInstructions.back()));
}

ConstantLayer_impl::~ConstantLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

void ConstantLayer_impl::setWeights(Weights weights) noexcept
{
    pass_warning("TODO! implement me!", true);
}

Weights ConstantLayer_impl::getWeights() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return Weights{};
}

void ConstantLayer_impl::setDimensions(Dims const& dimensions) noexcept
{
    pass_warning("TODO! implement me!", true);
}

Dims ConstantLayer_impl::getDimensions() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return Dims{};
}

} // namespace nvinfer1
