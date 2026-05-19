#ifndef RECURRENCE_LAYER_IMPL_HPP
#define RECURRENCE_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    class RecurrenceLayer_impl : public IRecurrenceLayer, public apiv::VRecurrenceLayer, virtual public LoopBoundaryLayer_impl
    {
    public:
        RecurrenceLayer_impl();
        ~RecurrenceLayer_impl() override;

        void build() noexcept override;
    };

}  // namespace nvinfer1

#endif // RECURRENCE_LAYER_IMPL_HPP
