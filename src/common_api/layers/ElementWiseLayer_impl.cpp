// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/insert.hpp>

#include "ElementWiseLayer_impl.hpp"
#include "Helper.hpp"

namespace nvinfer1
{

ElementWiseLayer_impl::ElementWiseLayer_impl() noexcept
    : Layer_impl{LayerType::kELEMENTWISE, nullptr}, mOp{ElementWiseOperation::kSUM}
{
    IElementWiseLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));;
    IElementWiseLayer::mImpl = this;
}

ElementWiseLayer_impl::ElementWiseLayer_impl(ITensor& input1, ITensor& input2, ElementWiseOperation op, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kELEMENTWISE, program}, mOp{op}
{
    IElementWiseLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));;
    IElementWiseLayer::mImpl = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(input1));
    mInputs.push_back(&static_cast<Tensor_impl&>(input2));
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

ElementWiseLayer_impl::~ElementWiseLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

// public API
void ElementWiseLayer_impl::setOperation(ElementWiseOperation op) noexcept
{
    mOp = op;
}

ElementWiseOperation ElementWiseLayer_impl::getOperation() const noexcept
{
    return mOp;
}

void ElementWiseLayer_impl::build() noexcept
{
    auto args = getInputArguments();
    mInstructions.clear();
    mInstructions.push_back(args[0]);
    mInstructions.push_back(args[1]);

    auto* mm = getModule();

    const auto op_name = helper::toPointwiseOpName(mOp);

    mInstructions.push_back(migraphx::op::builder::add(op_name, *mm, args).at(0));

    mOutputs[0]->setInstruction(mInstructions.back());
}

}   // namespace nvinfer1
