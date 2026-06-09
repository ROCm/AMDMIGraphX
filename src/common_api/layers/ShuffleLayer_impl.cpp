#include <numeric>
#include <vector>

#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

#include "Helper.hpp"
#include "ShuffleLayer_impl.hpp"

namespace nvinfer1
{

ShuffleLayer_impl::ShuffleLayer_impl() noexcept
    : Layer_impl{LayerType::kSHUFFLE, nullptr}
{
    IShuffleLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IShuffleLayer::mImpl  = this;
}

ShuffleLayer_impl::ShuffleLayer_impl(ITensor& input, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kSHUFFLE, program}
{
    IShuffleLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    IShuffleLayer::mImpl  = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(input));
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

ShuffleLayer_impl::~ShuffleLayer_impl() = default;

// public API
void ShuffleLayer_impl::setFirstTranspose(Permutation const& permutation) noexcept
{
    mFirstTranspose    = permutation;
    mHasFirstTranspose = true;
}

Permutation const& ShuffleLayer_impl::getFirstTranspose() const noexcept { return mFirstTranspose; }

void ShuffleLayer_impl::setReshapeDimensions(Dims const& dimensions) noexcept
{
    mReshapeDimensions = dimensions;
    mHasReshape        = true;
}

Dims ShuffleLayer_impl::getReshapeDimensions() const noexcept { return mReshapeDimensions; }

void ShuffleLayer_impl::setSecondTranspose(Permutation const& permutation) noexcept
{
    mSecondTranspose    = permutation;
    mHasSecondTranspose = true;
}

Permutation const& ShuffleLayer_impl::getSecondTranspose() const noexcept { return mSecondTranspose; }

void ShuffleLayer_impl::setZeroIsPlaceholder(bool zeroIsPlaceholder) noexcept { mZeroIsPlaceholder = zeroIsPlaceholder; }

bool ShuffleLayer_impl::getZeroIsPlaceholder() const noexcept { return mZeroIsPlaceholder; }

void ShuffleLayer_impl::build() noexcept
{
    auto args = getInputArguments();
    mInstructions.clear();

    auto* mm   = getModule();
    auto result = args[0];
    mInstructions.push_back(result);

    // TensorRT IShuffleLayer applies, in order: first transpose, reshape, second
    // transpose. Each stage is optional and only emitted when configured. The
    // Permutation is applied directly as the MIGraphX transpose permutation
    // (output axis i is taken from input axis permutation[i]).
    if(mHasFirstTranspose)
    {
        const auto rank = static_cast<int64_t>(result->get_shape().ndim());
        std::vector<int64_t> perm(mFirstTranspose.order, mFirstTranspose.order + rank);
        result = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), result);
    }

    if(mHasReshape)
    {
        // Resolve TensorRT reshape placeholders: 0 means "copy the corresponding
        // input dimension" (when zeroIsPlaceholder is set), -1 means "infer".
        const auto cur_lens = result->get_shape().lens();
        std::vector<int64_t> dims;
        dims.reserve(mReshapeDimensions.nbDims);
        for(int32_t i = 0; i < mReshapeDimensions.nbDims; ++i)
        {
            int64_t d = mReshapeDimensions.d[i];
            if(d == 0 and mZeroIsPlaceholder and static_cast<std::size_t>(i) < cur_lens.size())
                d = static_cast<int64_t>(cur_lens[i]);
            dims.push_back(d);
        }
        // A preceding transpose yields a non-standard (strided) view; reshape needs
        // a packed layout, so materialize it first.
        if(not result->get_shape().standard())
            result = mm->add_instruction(migraphx::make_op("contiguous"), result);
        result = mm->add_instruction(migraphx::make_op("reshape", {{"dims", dims}}), result);
    }

    if(mHasSecondTranspose)
    {
        const auto rank = static_cast<int64_t>(result->get_shape().ndim());
        std::vector<int64_t> perm(mSecondTranspose.order, mSecondTranspose.order + rank);
        result = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), result);
    }

    mInstructions.push_back(result);
    mOutputs[0]->setInstruction(result);
}

}  // namespace nvinfer1
