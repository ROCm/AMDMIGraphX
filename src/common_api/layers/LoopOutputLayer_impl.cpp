#include "LoopOutputLayer_impl.hpp"
#include "Tensor_impl.hpp"

namespace nvinfer1
{

LoopOutputLayer_impl::LoopOutputLayer_impl(ITensor& tensor,
                                           LoopOutput outputKind,
                                           int32_t axis,
                                           const std::shared_ptr<migraphx::program>& program,
                                           ILoop* loop) noexcept
    : Layer_impl{LayerType::kLOOP_OUTPUT, program}, LoopBoundaryLayer_impl{loop}, mKind{outputKind}, mAxis{axis}
{
    ILoopOutputLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    ILoopOutputLayer::mImpl  = this;
    mBoundary                = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(tensor));
    // output 0 is bound to the loop result by Loop_impl::finalize().
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

LoopOutputLayer_impl::~LoopOutputLayer_impl() = default;

LoopOutput LoopOutputLayer_impl::getLoopOutput() const noexcept
{
    return mKind;
}

void LoopOutputLayer_impl::setAxis(int32_t axis) noexcept
{
    mAxis = axis;
}

int32_t LoopOutputLayer_impl::getAxis() const noexcept
{
    return mAxis;
}

void LoopOutputLayer_impl::setInput(int32_t index, ITensor& tensor) noexcept
{
    auto* tensorImpl = dynamic_cast<Tensor_impl*>(&tensor);
    if(index == static_cast<int32_t>(mInputs.size()))
    {
        mInputs.push_back(tensorImpl);
    }
    else
    {
        mInputs.at(index) = tensorImpl;
    }
}

}  // namespace nvinfer1
