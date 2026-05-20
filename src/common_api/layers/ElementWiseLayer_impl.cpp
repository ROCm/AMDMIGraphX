// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "ElementWiseLayer_impl.hpp"

namespace nvinfer1
{
    
ElementWiseLayer_impl::ElementWiseLayer_impl() noexcept
{

}

ElementWiseLayer_impl::ElementWiseLayer_impl(ITensor& input1, ITensor& input2, ElementWiseOperation op, const std::shared_ptr<migraphx::program>& program) noexcept
{

}

ElementWiseLayer_impl::~ElementWiseLayer_impl() noexcept
{

}

// public API
void ElementWiseLayer_impl::setOperation(ElementWiseOperation op) noexcept
{
    pass_warning("TODO! implement me!", true);
}

ElementWiseOperation ElementWiseLayer_impl::getOperation() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return ElementWiseOperation::kSUM;
}

void ElementWiseLayer_impl::build() noexcept
{
    pass_warning("TODO! implement me!", true);
}

}   // namespace nvinfer1
