// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "IteratorLayer_impl.hpp"

namespace nvinfer1
{
IteratorLayer_impl::IteratorLayer_impl()
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

IteratorLayer_impl::~IteratorLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

void IteratorLayer_impl::setAxis(int32_t axis) noexcept
{
    pass_warning("TODO! implement me!", true);
}

int32_t IteratorLayer_impl::getAxis() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void IteratorLayer_impl::setReverse(bool reverse) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool IteratorLayer_impl::getReverse() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void IteratorLayer_impl::build() noexcept
{
    pass_warning("TODO! implement me!", true);
}

}   // namespace nvinfer1
