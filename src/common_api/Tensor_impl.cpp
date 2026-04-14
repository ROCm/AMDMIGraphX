#include "Tensor_impl.hpp"

namespace nvinfer1
{

Tensor_impl::Tensor_impl() noexcept
{
    // TODO! implement
    mImpl = this;
}

Tensor_impl::Tensor_impl(migraphx::instruction_ref ins) noexcept : mIns{ins}
{
    mImpl = this;
}

Tensor_impl::~Tensor_impl()
{
    // TODO! implement
}

void Tensor_impl::setName(char const* name) noexcept
{
    // TODO! implement
}

char const* Tensor_impl::getName() const noexcept
{
    // TODO! implement
    return nullptr;
}

void Tensor_impl::setDimensions(Dims const& dimensions) noexcept
{
    // TODO! implement
}

Dims Tensor_impl::getDimensions() const noexcept
{
    // TODO! implement
    return Dims{};
}

void Tensor_impl::setType(DataType type) noexcept
{
    // TODO! implement
}

DataType Tensor_impl::getType() const noexcept
{
    // TODO! implement
    return DataType::kFLOAT;
}

bool Tensor_impl::setDynamicRange(float min, float max) noexcept
{
    // TODO! implement
    return false;
}

bool Tensor_impl::isNetworkInput() const noexcept
{
    // TODO! implement
    return false;
}

bool Tensor_impl::isNetworkOutput() const noexcept
{
    // TODO! implement
    return false;
}

void Tensor_impl::setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept
{
    // TODO! implement
}

bool Tensor_impl::getBroadcastAcrossBatch() const noexcept
{
    // TODO! implement
    return false;
}

TensorLocation Tensor_impl::getLocation() const noexcept
{
    // TODO! implement
    return TensorLocation::kDEVICE;
}

void Tensor_impl::setLocation(TensorLocation location) noexcept
{
    // TODO! implement
}

bool Tensor_impl::dynamicRangeIsSet() const noexcept
{
    // TODO! implement
    return false;
}

void Tensor_impl::resetDynamicRange() noexcept
{
    // TODO! implement
}

float Tensor_impl::getDynamicRangeMin() const noexcept
{
    // TODO! implement
    return 0.0f;
}

float Tensor_impl::getDynamicRangeMax() const noexcept
{
    // TODO! implement
    return 0.0f;
}

void Tensor_impl::setAllowedFormats(TensorFormats formats) noexcept
{
    // TODO! implement
}

TensorFormats Tensor_impl::getAllowedFormats() const noexcept
{
    // TODO! implement
    return 0;
}

bool Tensor_impl::isShapeTensor() const noexcept
{
    // TODO! implement
    return false;
}

bool Tensor_impl::isExecutionTensor() const noexcept
{
    // TODO! implement
    return false;
}

void Tensor_impl::setDimensionName(int32_t index, char const* name) noexcept
{
    // TODO! implement
}

char const* Tensor_impl::getDimensionName(int32_t index) const noexcept
{
    // TODO! implement
    return nullptr;
}


}  // ns:nvinfer1
