// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "LoopOutputLayer_impl.hpp"

namespace nvinfer1
{
LoopOutputLayer_impl::LoopOutputLayer_impl()
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

LoopOutputLayer_impl::~LoopOutputLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

LoopOutput LoopOutputLayer_impl::getLoopOutput() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return LoopOutput::kLAST_VALUE;
}

void LoopOutputLayer_impl::setAxis(int32_t axis) noexcept
{
    pass_warning("TODO! implement me!", true);
}

int32_t LoopOutputLayer_impl::getAxis() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void LoopOutputLayer_impl::build() noexcept
{
    pass_warning("TODO! implement me!", true);
}

}  // namespace nvinfer1
