// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "Tensor_impl.hpp"
#include "Helper.hpp"

namespace nvinfer1
{

Tensor_impl::Tensor_impl() noexcept
    : mName{""}
{
    mImpl = this;
}

Tensor_impl::Tensor_impl(migraphx::instruction_ref ins) noexcept 
    : mIns{ins}, mName{ins->name()}, mBound{true}
{
    mImpl = this;
}

Tensor_impl::~Tensor_impl()
{
    pass_warning("TODO! implement me!", false);
}

migraphx::instruction_ref Tensor_impl::getInstruction() const noexcept
{
    return mIns;
}

void Tensor_impl::setInstruction(migraphx::instruction_ref ins) noexcept 
{
    mIns = ins;
    mBound = true;
}

void Tensor_impl::setName(char const* name) noexcept
{
    // pass_warning("TODO! implement me!", true);
    mName = name;
}

char const* Tensor_impl::getName() const noexcept
{
    // pass_warning("TODO! implement me!", true);
    return mName.c_str();
}

void Tensor_impl::setDimensions(Dims const& dimensions) noexcept
{
    pass_warning("TODO! implement me!", true);
}

Dims Tensor_impl::getDimensions() const noexcept
{
    if(!mBound) return Dims{};
    return helper::toDimensions(mIns->get_shape());
}

void Tensor_impl::setType(DataType type) noexcept
{
    mType    = type;
    mTypeSet = true;
}

DataType Tensor_impl::getType() const noexcept
{
    // Once the tensor is wired into the graph its data type is whatever the
    // producing instruction yields; otherwise fall back to any type the caller
    // explicitly requested via setType().
    if(mBound)
        return helper::toDataType(mIns->get_shape().type());
    return mType;
}

bool Tensor_impl::setDynamicRange(float min, float max) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool Tensor_impl::isNetworkInput() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool Tensor_impl::isNetworkOutput() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void Tensor_impl::setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool Tensor_impl::getBroadcastAcrossBatch() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

TensorLocation Tensor_impl::getLocation() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return TensorLocation::kDEVICE;
}

void Tensor_impl::setLocation(TensorLocation location) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool Tensor_impl::dynamicRangeIsSet() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void Tensor_impl::resetDynamicRange() noexcept
{
    pass_warning("TODO! implement me!", true);
}

float Tensor_impl::getDynamicRangeMin() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0.0f;
}

float Tensor_impl::getDynamicRangeMax() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0.0f;
}

void Tensor_impl::setAllowedFormats(TensorFormats formats) noexcept
{
    pass_warning("TODO! implement me!", true);
}

TensorFormats Tensor_impl::getAllowedFormats() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool Tensor_impl::isShapeTensor() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool Tensor_impl::isExecutionTensor() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void Tensor_impl::setDimensionName(int32_t index, char const* name) noexcept
{
    pass_warning("TODO! implement me!", true);
}

char const* Tensor_impl::getDimensionName(int32_t index) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}


}  // ns:nvinfer1
