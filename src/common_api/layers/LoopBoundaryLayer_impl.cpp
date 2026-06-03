#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{

LoopBoundaryLayer_impl::LoopBoundaryLayer_impl() = default;

LoopBoundaryLayer_impl::LoopBoundaryLayer_impl(ILoop* loop)
    : mLoop{loop}
{
}

LoopBoundaryLayer_impl::~LoopBoundaryLayer_impl() = default;

ILoop* LoopBoundaryLayer_impl::getLoop() const noexcept
{
    return mLoop;
}

void LoopBoundaryLayer_impl::build() noexcept
{
    // Intentionally empty. Loop_impl drives the construction of the
    // migraphx loop instruction and its submodule.
}

} // namespace nvinfer1
