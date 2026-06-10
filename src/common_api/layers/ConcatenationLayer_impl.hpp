#ifndef CONCATENATION_LAYER_IMPL_HPP
#define CONCATENATION_LAYER_IMPL_HPP

#include <vector>

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class ConcatenationLayer_impl : public IConcatenationLayer, public apiv::VConcatenationLayer, virtual public Layer_impl
    {
    public:
        ConcatenationLayer_impl() noexcept;
        ConcatenationLayer_impl(const std::vector<ITensor*>& inputs, int axis, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~ConcatenationLayer_impl() override;

        // public API
        void setAxis(int axis) noexcept override;
        int getAxis() const noexcept override;

        void build() noexcept override;

    private:
        int mAxis{0};
    };
} // namespace nvinfer1

#endif // CONCATENATION_LAYER_IMPL_HPP
