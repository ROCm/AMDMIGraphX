#ifndef LOOP_BOUNDARY_LAYER_IMPL_HPP
#define LOOP_BOUNDARY_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "Layer_impl.hpp"

namespace nvinfer1
{
    class LoopBoundaryLayer_impl : public ILoopBoundaryLayer, public apiv::VLoopBoundaryLayer, virtual public Layer_impl
    {
    public:
        LoopBoundaryLayer_impl();
        ~LoopBoundaryLayer_impl() override;

        // public API
        ILoop* getLoop() const noexcept override;
    };

}  // namespace nvinfer1

#endif // LOOP_BOUNDARY_LAYER_IMPL_HPP
