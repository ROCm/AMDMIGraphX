#include "RecurrenceLayer_impl.hpp"
#include "Tensor_impl.hpp"

namespace nvinfer1
{

RecurrenceLayer_impl::RecurrenceLayer_impl(ITensor& initialValue,
                                           const std::shared_ptr<migraphx::program>& program,
                                           ILoop* loop) noexcept
    : Layer_impl{LayerType::kRECURRENCE, program}, LoopBoundaryLayer_impl{loop}
{
    IRecurrenceLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IRecurrenceLayer::mImpl  = this;
    mBoundary                = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(initialValue));
    // output 0 is a placeholder; Loop_impl::preBuild() binds it to the loop
    // body parameter that carries this recurrence's value.
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

RecurrenceLayer_impl::~RecurrenceLayer_impl() = default;

void RecurrenceLayer_impl::setInput(int32_t index, ITensor& tensor) noexcept
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
