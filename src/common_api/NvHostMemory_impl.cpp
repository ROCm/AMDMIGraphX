#include "NvHostMemory_impl.hpp"

namespace nvinfer1
{

NvHostMemory_impl::NvHostMemory_impl(void* data, size_t size) noexcept
{
    // TODO! implement
    mImpl = this;
}

NvHostMemory_impl::~NvHostMemory_impl()
{
    // TODO! implement
}

// public API
void* NvHostMemory_impl::data() const noexcept
{
    // TODO! implement
    return nullptr;
}

size_t NvHostMemory_impl::size() const noexcept
{
    // TODO! implement
    return 0;
}

}   // ns:nvinfer1
