#include <numeric>
#include <vector>
#include <cstdint>

#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/insert.hpp>

#include "Helper.hpp"
#include "SliceLayer_impl.hpp"

namespace nvinfer1
{

enum class InputIndex : int32_t
{
    kData   = 0,
    kStart  = 1,
    kSize   = 2,
    kStride = 3,
    kFill   = 4,
    kAxes   = 5,
};

SliceLayer_impl::SliceLayer_impl() noexcept
    : Layer_impl{LayerType::kSLICE, nullptr}
{
    ISliceLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    ISliceLayer::mImpl  = this;
}

SliceLayer_impl::SliceLayer_impl(ITensor& input, Dims const& start, Dims const& size, Dims const& stride, const std::shared_ptr<migraphx::program>& program) noexcept
    : Layer_impl{LayerType::kSLICE, program}, mStart{start}, mSize{size}, mStride{stride}
{
    ISliceLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    ISliceLayer::mImpl  = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(input));
    mOutputs.emplace_back(std::make_unique<Tensor_impl>());
}

SliceLayer_impl::~SliceLayer_impl() = default;

// public API
void SliceLayer_impl::setStart(Dims const& start) noexcept { mStart = start; }
Dims SliceLayer_impl::getStart() const noexcept { return mStart; }

void SliceLayer_impl::setSize(Dims const& size) noexcept { mSize = size; }
Dims SliceLayer_impl::getSize() const noexcept { return mSize; }

void SliceLayer_impl::setStride(Dims const& stride) noexcept { mStride = stride; }
Dims SliceLayer_impl::getStride() const noexcept { return mStride; }

void SliceLayer_impl::setMode(SampleMode mode) noexcept { mMode = mode; }
SampleMode SliceLayer_impl::getMode() const noexcept { return mMode; }

void SliceLayer_impl::setAxes(Dims const& axes) noexcept { mAxes = axes; }
Dims SliceLayer_impl::getAxes() const noexcept { return mAxes; }

void SliceLayer_impl::setInput(int32_t index, ITensor& tensor) noexcept
{
    auto* tensorImpl = dynamic_cast<Tensor_impl*>(&tensor);
    switch(static_cast<InputIndex>(index))
    {
    case InputIndex::kData:
        if(mInputs.empty())
            mInputs.push_back(tensorImpl);
        else
            mInputs[0] = tensorImpl;
        break;
    case InputIndex::kFill:
        // Fill value for kFILL mode (a scalar produced by an IConstantLayer).
        mFill = tensorImpl;
        break;

    case InputIndex::kStart:
    case InputIndex::kSize:
    case InputIndex::kStride:
    case InputIndex::kAxes:
        // Dynamic start/size/stride/axes inputs are not exercised by the sample.
        break;
    }
}

void SliceLayer_impl::build() noexcept
{
    auto args = getInputArguments();
    mInstructions.clear();

    auto* mm   = getModule();
    auto data  = args[static_cast<std::size_t>(InputIndex::kData)];
    mInstructions.push_back(data);

    const auto in_shape          = data->get_shape();
    const std::vector<std::size_t> in_lens = in_shape.lens();
    const auto rank              = static_cast<int64_t>(in_lens.size());

    // Resolve which axes the start/size/stride entries apply to. When no axes are
    // supplied (the common case) they map one-to-one onto every input dimension.
    std::vector<int64_t> axes;
    if(mAxes.nbDims > 0)
    {
        axes = helper::dimsToVec(mAxes);
        for(auto& a : axes)
            if(a < 0)
                a += rank;
    }
    else
    {
        axes.resize(rank);
        std::iota(axes.begin(), axes.end(), 0);
    }

    const auto start_vec  = helper::dimsToVec(mStart);
    const auto size_vec   = helper::dimsToVec(mSize);
    const auto stride_vec = helper::dimsToVec(mStride);

    // Per-input-dimension slice parameters. Axes not listed keep the full range.
    std::vector<int64_t> ax_start(rank, 0);
    std::vector<int64_t> ax_stride(rank, 1);
    std::vector<int64_t> out_lens(in_lens.begin(), in_lens.end());
    for(std::size_t i = 0; i < axes.size(); ++i)
    {
        const auto axis = axes[i];
        ax_start[axis]  = start_vec[i];
        out_lens[axis]  = size_vec[i];
        ax_stride[axis] = stride_vec[i];
    }

    // Row-major strides used to flatten an input coordinate into a linear index.
    std::vector<int64_t> in_strides(rank, 1);
    for(int64_t d = rank - 2; d >= 0; --d)
        in_strides[d] = in_strides[d + 1] * static_cast<int64_t>(in_lens[d + 1]);

    const int64_t total_out = std::accumulate(out_lens.begin(), out_lens.end(), int64_t{1}, std::multiplies<int64_t>());
    const int64_t total_in  = std::accumulate(in_lens.begin(), in_lens.end(), int64_t{1}, std::multiplies<int64_t>());

    // Map an out-of-(or in-)range coordinate on one axis to a valid input index,
    // following TensorRT SampleMode semantics. Returns {index, out_of_bounds}.
    auto map_coord = [&](int64_t coord, int64_t len) -> std::pair<int64_t, bool> {
        switch(mMode)
        {
        case SampleMode::kWRAP:
        {
            const int64_t m = ((coord % len) + len) % len;
            return {m, false};
        }
        case SampleMode::kCLAMP:
        {
            const int64_t c = coord < 0 ? 0 : (coord >= len ? len - 1 : coord);
            return {c, false};
        }
        case SampleMode::kREFLECT:
        {
            if(len == 1)
                return {0, false};
            const int64_t period = 2 * (len - 1);
            const int64_t m      = ((coord % period) + period) % period;
            return {m < len ? m : period - m, false};
        }
        case SampleMode::kFILL:
        {
            const bool oob = (coord < 0 or coord >= len);
            return {oob ? 0 : coord, oob};
        }
        case SampleMode::kSTRICT_BOUNDS:
            return {coord, false};
        }
    };

    // Row-major strides over the output shape, for decoding a flat output index
    // back into per-axis coordinates.
    std::vector<int64_t> out_strides(rank, 1);
    for(int64_t d = rank - 2; d >= 0; --d)
        out_strides[d] = out_strides[d + 1] * out_lens[d + 1];

    // Enumerate every output element (row-major), compute the source coordinate on
    // each axis, remap per the slice mode and collapse to a flat input index.
    std::vector<int64_t> indices(total_out);
    std::vector<uint8_t> fill_mask(total_out, 0);
    bool any_fill = false;
    for(int64_t p = 0; p < total_out; ++p)
    {
        int64_t flat = 0;
        bool oob_any = false;
        for(int64_t d = 0; d < rank; ++d)
        {
            const int64_t o     = (p / out_strides[d]) % out_lens[d];
            const int64_t source = ax_start[d] + o * ax_stride[d];
            const auto mapped   = map_coord(source, static_cast<int64_t>(in_lens[d]));
            flat += mapped.first * in_strides[d];
            oob_any = oob_any or mapped.second;
        }
        indices[p]   = flat;
        fill_mask[p] = oob_any ? 1 : 0;
        any_fill     = any_fill or oob_any;
    }

    // Flatten the input and gather the precomputed source indices.
    auto flat_data = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {total_in}}}), data);

    migraphx::shape idx_shape{migraphx::shape::int64_type, {static_cast<std::size_t>(total_out)}};
    auto idx_lit = mm->add_literal(idx_shape, reinterpret_cast<const uint8_t*>(indices.data()));

    auto gathered = migraphx::op::builder::add("gather", *mm, {flat_data, idx_lit}, {{"axis", 0}}).at(0);

    auto result = mm->add_instruction(migraphx::make_op("reshape", {{"dims", out_lens}}), gathered);

    // For kFILL, replace the out-of-bounds positions with the supplied fill value.
    if(mMode == SampleMode::kFILL and any_fill and mFill != nullptr)
    {
        migraphx::shape mask_shape{migraphx::shape::bool_type, std::vector<std::size_t>(out_lens.begin(), out_lens.end())};
        auto mask_lit = mm->add_literal(mask_shape, reinterpret_cast<const uint8_t*>(fill_mask.data()));

        auto fill_val   = mFill->getInstruction();
        auto fill_bcast = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), fill_val);

        result = mm->add_instruction(migraphx::make_op("where"), mask_lit, fill_bcast, result);
    }

    mInstructions.push_back(result);
    mOutputs[0]->setInstruction(result);
}

}  // namespace nvinfer1
