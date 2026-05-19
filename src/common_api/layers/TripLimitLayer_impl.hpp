#ifndef TRIP_LIMIT_LAYER_IMPL_HPP
#define TRIP_LIMIT_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    class TripLimitLayer_impl : public ITripLimitLayer, public apiv::VTripLimitLayer, virtual public LoopBoundaryLayer_impl
    {
    public:
        TripLimitLayer_impl();
        ~TripLimitLayer_impl() override;

        // public API
        TripLimit getTripLimit() const noexcept override;

        void build() noexcept override;
    };

}  // namespace nvinfer1

#endif // TRIP_LIMIT_LAYER_IMPL_HPP
