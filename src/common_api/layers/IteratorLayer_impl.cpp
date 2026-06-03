#include "IteratorLayer_impl.hpp"
#include "Tensor_impl.hpp"

namespace nvinfer1
{

IteratorLayer_impl::IteratorLayer_impl(ITensor& tensor,
                                       int32_t axis,
                                       bool reverse,
                                       const std::shared_ptr<migraphx::program>& program,
                                       ILoop* loop) noexcept
    : Layer_impl{LayerType::kITERATOR, program}, LoopBoundaryLayer_impl{loop}, mAxis{axis}, mReverse{reverse}
{
    IIteratorLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IIteratorLayer::mImpl  = this;
    mBoundary              = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(tensor));
    // output 0 is bound to the per-iteration slice by Loop_impl::preBuild().
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

IteratorLayer_impl::~IteratorLayer_impl() = default;

void IteratorLayer_impl::setAxis(int32_t axis) noexcept
{
    mAxis = axis;
}

int32_t IteratorLayer_impl::getAxis() const noexcept
{
    return mAxis;
}

void IteratorLayer_impl::setReverse(bool reverse) noexcept
{
    mReverse = reverse;
}

bool IteratorLayer_impl::getReverse() const noexcept
{
    return mReverse;
}

}   // namespace nvinfer1
