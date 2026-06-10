#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

#include "ConcatenationLayer_impl.hpp"

namespace nvinfer1
{

ConcatenationLayer_impl::ConcatenationLayer_impl() noexcept
    : Layer_impl{LayerType::kCONCATENATION, nullptr}, mAxis{0}
{
    IConcatenationLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IConcatenationLayer::mImpl = this;
}

ConcatenationLayer_impl::ConcatenationLayer_impl(const std::vector<ITensor*>& inputs, int axis, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kCONCATENATION, program}, mAxis{axis}
{
    IConcatenationLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IConcatenationLayer::mImpl = this;

    for (auto* input : inputs)
    {
        mInputs.push_back(static_cast<Tensor_impl*>(input));
    }
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

ConcatenationLayer_impl::~ConcatenationLayer_impl()
{
}

void ConcatenationLayer_impl::setAxis(int axis) noexcept
{
    mAxis = axis;
}

int ConcatenationLayer_impl::getAxis() const noexcept
{
    return mAxis;
}

void ConcatenationLayer_impl::build() noexcept
{
    auto args = getInputArguments();

    mInstructions.clear();
    for (const auto& arg : args)
    {
        mInstructions.push_back(arg);
    }

    auto* mm = getModule();

    auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", mAxis}}), args);
    mInstructions.push_back(concat);

    mOutputs[0]->setInstruction(mInstructions.back());
}

}  // namespace nvinfer1
