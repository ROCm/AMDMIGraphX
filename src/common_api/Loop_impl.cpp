// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "Loop_impl.hpp"

namespace nvinfer1
{

Loop_impl::Loop_impl()
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

Loop_impl::~Loop_impl()
{
    pass_warning("TODO! implement me!", false);
}

// public API
IRecurrenceLayer* Loop_impl::addRecurrence(ITensor& initialValue) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

ITripLimitLayer* Loop_impl::addTripLimit(ITensor& tensor, TripLimit limit) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

IIteratorLayer* Loop_impl::addIterator(ITensor& tensor, int32_t axis /*= 0*/, bool reverse /*= false*/) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

ILoopOutputLayer* Loop_impl::addLoopOutput(ITensor& tensor, LoopOutput outputKind, int32_t axis /*= 0*/) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void Loop_impl::setName(char const* name) noexcept
{
    pass_warning("TODO! implement me!", true);
}

char const* Loop_impl::getName() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

} // namespace nvinfer1
