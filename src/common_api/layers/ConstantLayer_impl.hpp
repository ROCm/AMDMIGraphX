#ifndef CONSTANT_LAYER_IMPL_HPP
#define CONSTANT_LAYER_IMPL_HPP 

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class ConstantLayer_impl : public IConstantLayer, public apiv::VConstantLayer, virtual public Layer_impl
    {
    public:
        ConstantLayer_impl() noexcept;
        ConstantLayer_impl(Dims const& dimensions, Weights weights, const std::shared_ptr<migraphx::program>& program) noexcept;

        ~ConstantLayer_impl() override;

        // public API
        void setWeights(Weights weights) noexcept override;
        Weights getWeights() const noexcept override;
        void setDimensions(Dims const& dimensions) noexcept override;
        Dims getDimensions() const noexcept override;

        void build() noexcept override;

    private:
        Dims mDimensions;
        Weights mWeights;
    };

} // namespace nvinfer1

#endif // CONSTANT_LAYER_IMPL_HPP
