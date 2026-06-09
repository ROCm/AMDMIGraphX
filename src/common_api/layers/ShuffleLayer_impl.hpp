#ifndef SHUFFLE_LAYER_IMPL_HPP
#define SHUFFLE_LAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class ShuffleLayer_impl : public IShuffleLayer, public apiv::VShuffleLayer, virtual public Layer_impl
    {
    public:
        ShuffleLayer_impl() noexcept;
        ShuffleLayer_impl(ITensor& input, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~ShuffleLayer_impl() override;

        // public API
        void setFirstTranspose(Permutation const& permutation) noexcept override;
        Permutation const& getFirstTranspose() const noexcept override;
        void setReshapeDimensions(Dims const& dimensions) noexcept override;
        Dims getReshapeDimensions() const noexcept override;
        void setSecondTranspose(Permutation const& permutation) noexcept override;
        Permutation const& getSecondTranspose() const noexcept override;
        void setZeroIsPlaceholder(bool zeroIsPlaceholder) noexcept override;
        bool getZeroIsPlaceholder() const noexcept override;

        void build() noexcept override;

    private:
        Permutation mFirstTranspose{};
        Permutation mSecondTranspose{};
        Dims mReshapeDimensions{};
        bool mHasFirstTranspose{false};
        bool mHasReshape{false};
        bool mHasSecondTranspose{false};
        bool mZeroIsPlaceholder{true};
    };
}

#endif // SHUFFLE_LAYER_IMPL_HPP
