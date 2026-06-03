#ifndef ITERATOR_LAYER_IMPL_HPP
#define ITERATOR_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    //! Iterates over an input tensor along an axis, exposing one slice per
    //! iteration. Equivalent to gather(tensor, I, axis) where I is the (possibly
    //! reversed) loop iteration index.
    class IteratorLayer_impl : public IIteratorLayer, public apiv::VIteratorLayer, virtual public LoopBoundaryLayer_impl
    {
    public:
        IteratorLayer_impl(ITensor& tensor,
                           int32_t axis,
                           bool reverse,
                           const std::shared_ptr<migraphx::program>& program,
                           ILoop* loop) noexcept;
        ~IteratorLayer_impl() override;

        // public API
        void setAxis(int32_t axis) noexcept override;
        int32_t getAxis() const noexcept override;
        void setReverse(bool reverse) noexcept override;
        bool getReverse() const noexcept override;

    private:
        int32_t mAxis;
        bool mReverse;
    };

}  // namespace nvinfer1

#endif // ITERATOR_LAYER_IMPL_HPP
