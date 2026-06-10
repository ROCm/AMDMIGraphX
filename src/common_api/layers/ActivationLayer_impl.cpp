#include <vector>

#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/builder/insert.hpp>

#include "ActivationLayer_impl.hpp"

namespace nvinfer1
{

ActivationLayer_impl::ActivationLayer_impl() noexcept
    : Layer_impl{LayerType::kACTIVATION, nullptr}
{
    IActivationLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IActivationLayer::mImpl  = this;
}

ActivationLayer_impl::ActivationLayer_impl(ITensor& input, ActivationType type, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kACTIVATION, program}, mActivationType{type}
{
    IActivationLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IActivationLayer::mImpl  = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(input));
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

ActivationLayer_impl::~ActivationLayer_impl() = default;

// public API
void ActivationLayer_impl::setActivationType(ActivationType type) noexcept { mActivationType = type; }
ActivationType ActivationLayer_impl::getActivationType() const noexcept { return mActivationType; }

void ActivationLayer_impl::setAlpha(float alpha) noexcept { mAlpha = alpha; }
float ActivationLayer_impl::getAlpha() const noexcept { return mAlpha; }

void ActivationLayer_impl::setBeta(float beta) noexcept { mBeta = beta; }
float ActivationLayer_impl::getBeta() const noexcept { return mBeta; }

void ActivationLayer_impl::build() noexcept
{
    auto args = getInputArguments();
    mInstructions.clear();

    auto* mm = getModule();
    auto x   = args[0];
    mInstructions.push_back(x);

    const auto lens = x->get_shape().lens();

    // Materialize a scalar constant broadcast to the input shape, so it can be
    // combined element-wise with x.
    auto scalar = [&](float v) {
        auto lit = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {v}});
        return mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), lit);
    };

    migraphx::instruction_ref result = x;
    switch(mActivationType)
    {
    case ActivationType::kRELU:
        result = mm->add_instruction(migraphx::make_op("relu"), x);
        break;
    case ActivationType::kSIGMOID:
        result = mm->add_instruction(migraphx::make_op("sigmoid"), x);
        break;
    case ActivationType::kTANH:
        result = mm->add_instruction(migraphx::make_op("tanh"), x);
        break;
    case ActivationType::kLEAKY_RELU:
        result = mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", mAlpha}}), x);
        break;
    case ActivationType::kCLIP:
        // Clip(x) = max(alpha, min(beta, x)).
        result = migraphx::op::builder::add("clip", *mm, {x, scalar(mAlpha), scalar(mBeta)}).at(0);
        break;
    case ActivationType::kHARD_SIGMOID:
    {
        // HardSigmoid(x) = max(0, min(1, alpha*x + beta)).
        auto ax  = mm->add_instruction(migraphx::make_op("mul"), x, scalar(mAlpha));
        auto axb = mm->add_instruction(migraphx::make_op("add"), ax, scalar(mBeta));
        result   = migraphx::op::builder::add("clip", *mm, {axb, scalar(0.0f), scalar(1.0f)}).at(0);
        break;
    }
    default:
        // Unsupported activation type: pass the input through unchanged.
        result = mm->add_instruction(migraphx::make_op("identity"), x);
        break;
    }

    mInstructions.push_back(result);
    mOutputs[0]->setInstruction(result);
}

} // namespace nvinfer1
