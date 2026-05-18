#ifndef GATHER_LAYER_IMPL_HPP
#define GATHER_LAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class GatherLayer_impl : public IGatherLayer, public apiv::VGatherLayer, virtual public Layer_impl
    {
    public:
        GatherLayer_impl() noexcept;
        GatherLayer_impl(ITensor& data, ITensor& indices, int32_t axis, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~GatherLayer_impl() override;

        // public API
        void setGatherAxis(int32_t axis) noexcept override;
        int32_t getGatherAxis() const noexcept override;
        void setNbElementWiseDims(int32_t k) noexcept override;
        int32_t getNbElementWiseDims() const noexcept override;
        void setMode(GatherMode mode) noexcept override;
        GatherMode getMode() const noexcept override;

        void build() noexcept override;

    private:
        GatherMode mMode;
        int32_t mAxis;
        int32_t mNbElementWiseDims;
    };
}

#endif // GATHER_LAYER_IMPL_HPP
