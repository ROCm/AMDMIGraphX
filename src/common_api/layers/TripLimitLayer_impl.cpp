#include "TripLimitLayer_impl.hpp"
#include "Tensor_impl.hpp"

namespace nvinfer1
{

TripLimitLayer_impl::TripLimitLayer_impl(ITensor& tensor,
                                         TripLimit limit,
                                         const std::shared_ptr<migraphx::program>& program,
                                         ILoop* loop) noexcept
    : Layer_impl{LayerType::kTRIP_LIMIT, program}, LoopBoundaryLayer_impl{loop}, mLimit{limit}
{
    ITripLimitLayer::mLayer = static_cast<VLayer*>(static_cast<Layer_impl*>(this));
    ITripLimitLayer::mImpl  = this;
    mBoundary               = this;

    mInputs.push_back(&static_cast<Tensor_impl&>(tensor));
}

TripLimitLayer_impl::~TripLimitLayer_impl() = default;

TripLimit TripLimitLayer_impl::getTripLimit() const noexcept
{
    return mLimit;
}

}   // namespace nvinfer1
