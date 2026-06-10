#ifndef ACTIVATION_LAYER_IMPL_HPP
#define ACTIVATION_LAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class ActivationLayer_impl : public IActivationLayer, public apiv::VActivationLayer, virtual public Layer_impl
    {
    public:
        ActivationLayer_impl() noexcept;
        ActivationLayer_impl(ITensor& input, ActivationType type, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~ActivationLayer_impl() override;

        // public API
        void setActivationType(ActivationType type) noexcept override;
        ActivationType getActivationType() const noexcept override;
        void setAlpha(float alpha) noexcept override;
        float getAlpha() const noexcept override;
        void setBeta(float beta) noexcept override;
        float getBeta() const noexcept override;

        void build() noexcept override;

    private:
        ActivationType mActivationType{ActivationType::kRELU};
        float mAlpha{0.0f};
        float mBeta{0.0f};
    };
}

#endif // ACTIVATION_LAYER_IMPL_HPP
