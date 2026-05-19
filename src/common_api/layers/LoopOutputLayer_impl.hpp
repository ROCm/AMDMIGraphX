#ifndef LOOP_OUTPUT_LAYER_IMPL_HPP
#define LOOP_OUTPUT_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    class LoopOutputLayer_impl : public ILoopOutputLayer, public apiv::VLoopOutputLayer, public LoopBoundaryLayer_impl
    {
    public:
        LoopOutputLayer_impl();
        ~LoopOutputLayer_impl() override;

        // public API
        LoopOutput getLoopOutput() const noexcept override;
        void setAxis(int32_t axis) noexcept override;
        int32_t getAxis() const noexcept override;

        void build() noexcept override;
    };

}  // namespace nvinfer1

#endif // LOOP_OUTPUT_LAYER_IMPL_HPP
