#include "NvHostMemory_impl.hpp"

namespace nvinfer1
{

NvHostMemory_impl::NvHostMemory_impl(void* data, size_t size, DataType type) noexcept
    : mData(data), mSize(size), mType(type)
{
    mImpl = this;
}

NvHostMemory_impl::~NvHostMemory_impl()
{
    // TODO! implement
}

// public API
void* NvHostMemory_impl::data() const noexcept
{
    return mData;
}

size_t NvHostMemory_impl::size() const noexcept
{
    return mSize;
}

DataType NvHostMemory_impl::type() const noexcept
{
    return mType;
}

}   // ns:nvinfer1
