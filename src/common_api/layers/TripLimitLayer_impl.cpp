// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "TripLimitLayer_impl.hpp"

namespace nvinfer1
{
TripLimitLayer_impl::TripLimitLayer_impl()
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

TripLimitLayer_impl::~TripLimitLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

TripLimit TripLimitLayer_impl::getTripLimit() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return TripLimit::kCOUNT;
}

void TripLimitLayer_impl::build() noexcept
{
    pass_warning("TODO! implement me!", false);
}

}   // namespace nvinfer1
