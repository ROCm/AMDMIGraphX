#ifndef SLICE_LAYER_IMPL_HPP
#define SLICE_LAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class SliceLayer_impl : public ISliceLayer, public apiv::VSliceLayer, virtual public Layer_impl
    {
    public:
        SliceLayer_impl() noexcept;
        SliceLayer_impl(ITensor& input, Dims const& start, Dims const& size, Dims const& stride, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~SliceLayer_impl() override;

        // public API
        void setStart(Dims const& start) noexcept override;
        Dims getStart() const noexcept override;
        void setSize(Dims const& size) noexcept override;
        Dims getSize() const noexcept override;
        void setStride(Dims const& stride) noexcept override;
        Dims getStride() const noexcept override;
        void setMode(SampleMode mode) noexcept override;
        SampleMode getMode() const noexcept override;
        void setAxes(Dims const& axes) noexcept override;
        Dims getAxes() const noexcept override;

        // ISliceLayer exposes optional dynamic inputs (start/size/stride/fill/axes)
        // through ILayer::setInput. The fill value (index 4) for kFILL mode is the
        // only one used by the sample, so route it to a dedicated slot instead of
        // the base-class argument-replacement logic.
        void setInput(int32_t index, ITensor& tensor) noexcept override;

        void build() noexcept override;

    private:
        Dims mStart{};
        Dims mSize{};
        Dims mStride{};
        Dims mAxes{};
        SampleMode mMode{SampleMode::kSTRICT_BOUNDS};
        Tensor_impl* mFill{nullptr};
    };
}

#endif // SLICE_LAYER_IMPL_HPP
