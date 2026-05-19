#ifndef ITERATOR_LAYER_IMPL_HPP
#define ITERATOR_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    class IteratorLayer_impl : public IIteratorLayer, public apiv::VIteratorLayer, public LoopBoundaryLayer_impl
    {
    public:
        IteratorLayer_impl();
        ~IteratorLayer_impl() override;

        // public API
        void setAxis(int32_t axis) noexcept override;
        int32_t getAxis() const noexcept override;
        void setReverse(bool reverse) noexcept override;
        bool getReverse() const noexcept override;

        void build() noexcept override;
    };

}  // namespace nvinfer1

#endif // ITERATOR_LAYER_IMPL_HPP
