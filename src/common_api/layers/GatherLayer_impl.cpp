// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/insert.hpp>

#include "GatherLayer_impl.hpp"
#include "Helper.hpp"

namespace nvinfer1
{

GatherLayer_impl::GatherLayer_impl() noexcept
    : Layer_impl{LayerType::kGATHER, nullptr}, mMode{GatherMode::kDEFAULT}, mAxis{0}, mNbElementWiseDims{0}
{
    IGatherLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));;
    IGatherLayer::mImpl = this;
}

GatherLayer_impl::GatherLayer_impl(ITensor& data, ITensor& indices, int32_t axis, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kGATHER, program}, mMode{GatherMode::kDEFAULT}, mAxis{axis}, mNbElementWiseDims{0}
{
    IGatherLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));;
    IGatherLayer::mImpl = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(data));
    mInputs.push_back(&static_cast<Tensor_impl&>(indices));
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

GatherLayer_impl::~GatherLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

void GatherLayer_impl::build() noexcept
{
    auto args = getInputArguments();
    mInstructions.clear();
    mInstructions.push_back(args[0]);
    mInstructions.push_back(args[1]);

    auto* mm = mProgram->get_main_module();
    switch(getMode())
    {
    case GatherMode::kDEFAULT:
        mInstructions.push_back(migraphx::op::builder::add("gather", *mm, args, {{"axis", mAxis}}).at(0));
        break;
    case GatherMode::kELEMENT:
        mInstructions.push_back(migraphx::op::builder::add("gather_elements", *mm, args, {{"axis", mAxis}}).at(0));
        break;
    case GatherMode::kND:
        mInstructions.push_back(migraphx::op::builder::add("gathernd", *mm, args, {{"batch_dims", mNbElementWiseDims}}).at(0));
        break;
    }

    mOutputs[0]->setInstruction(mInstructions.back());
}

void GatherLayer_impl::setGatherAxis(int32_t axis) noexcept
{
    // TODO! The axis must be less than the number of dimensions in the data input.
    mAxis = axis;
}

int32_t GatherLayer_impl::getGatherAxis() const noexcept
{
    return mAxis;
}

void GatherLayer_impl::setNbElementWiseDims(int32_t k) noexcept
{
    // TODO! sanity check
    mNbElementWiseDims = k;
}

int32_t GatherLayer_impl::getNbElementWiseDims() const noexcept
{
    return mNbElementWiseDims;
}

void GatherLayer_impl::setMode(GatherMode mode) noexcept
{
    mMode = mode;
}

GatherMode GatherLayer_impl::getMode() const noexcept
{
    return mMode;
}

} // namespace nvinfer1
