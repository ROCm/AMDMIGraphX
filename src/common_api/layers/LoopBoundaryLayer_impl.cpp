// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{

LoopBoundaryLayer_impl::LoopBoundaryLayer_impl()
{
    pass_warning("TODO! implement me!", false);
    mBoundary = this;
}

LoopBoundaryLayer_impl::~LoopBoundaryLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

ILoop* LoopBoundaryLayer_impl::getLoop() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

} // namespace nvinfer1
