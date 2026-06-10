// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "TopKLayer_impl.hpp"

namespace nvinfer1
{

TopKLayer_impl::TopKLayer_impl() noexcept
{
    pass_warning("TODO! implement me!", false);
}

TopKLayer_impl::~TopKLayer_impl()
{
    pass_warning("TODO! implement me!", false);
}

void TopKLayer_impl::setOperation(TopKOperation op) noexcept
{
    pass_warning("TODO! implement me!", true);
}

TopKOperation TopKLayer_impl::getOperation() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return TopKOperation::kMAX;
}

void TopKLayer_impl::setK(int32_t k) noexcept
{
    pass_warning("TODO! implement me!", true);
}

int32_t TopKLayer_impl::getK() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void TopKLayer_impl::setReduceAxes(uint32_t reduceAxes) noexcept
{
    pass_warning("TODO! implement me!", true);
}

uint32_t TopKLayer_impl::getReduceAxes() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void TopKLayer_impl::build() noexcept
{
    pass_warning("TODO! implement me!", true);
}

}  // namespace nvinfer1
