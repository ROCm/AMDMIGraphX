#ifndef NV_HOST_MEMORY_IMPL_H
#define NV_HOST_MEMORY_IMPL_H

#include "migraphx/common_api/NvInferRuntime.h"

namespace nvinfer1
{
    class NvHostMemory_impl : public IHostMemory, public apiv::VHostMemory
    {
    public:
        NvHostMemory_impl(void* data, size_t size, DataType type) noexcept;
        ~NvHostMemory_impl() override;
        
        // public API
        void* data() const noexcept override;
        size_t size() const noexcept override;
        DataType type() const noexcept override;

    private:
        void* mData;
        size_t mSize;
        DataType mType;
    };

} // ns:nvinfer1

#endif // NV_HOST_MEMORY_IMPL_H
