#pragma once

#include "common.hpp"

namespace mlinfer
{
    template <typename AllocFunc, typename FreeFunc>
    struct IBuffer : public INoCopy
    {
        AllocFunc alloc_fn;
        FreeFunc free_fn;
    };

    template <typename AllocFunc, typename FreeFunc>
    struct GenericBuffer : public IBuffer<AllocFunc, FreeFunc>
    {
        GenericBuffer()
            : size_in_bytes{0}, stride_in_bytes{0}, tensor_ptr{nullptr}
        {
        }

        explicit GenericBuffer(size_t size_in_bytes_, size_t stride_in_bytes_ = 0)
            : size_in_bytes{size_in_bytes_}, stride_in_bytes{stride_in_bytes_}
        {
            if (stride_in_bytes == 0)
            {
                stride_in_bytes = size_in_bytes;
            }
            this->alloc_fn(&tensor_ptr, size_in_bytes);
        }

        GenericBuffer(GenericBuffer &&buf)
            : size_in_bytes{buf.size_in_bytes}, stride_in_bytes{buf.stride_in_bytes}, tensor_ptr{buf.tensor_ptr}
        {
            buf.size_in_bytes = 0;
            buf.stride_in_bytes = 0;
            buf.tensor_ptr = nullptr;
        }

        GenericBuffer &operator=(GenericBuffer &&buf)
        {
            if (this != &buf)
            {
                this->free_fn(tensor_ptr);
                size_in_bytes = buf.size_in_bytes;
                stride_in_bytes = buf.stride_in_bytes;
                tensor_ptr = buf.tensor_ptr;
                buf.size_in_bytes = 0;
                buf.stride_in_bytes = 0;
                buf.tensor_ptr = nullptr;
            }
            return *this;
        }

        GenericBuffer(const GenericBuffer &buf) = delete;
        GenericBuffer &operator=(const GenericBuffer &buf) = delete;

        ~GenericBuffer()
        {
            this->free_fn(tensor_ptr);
        }

        size_t size_in_bytes;
        size_t stride_in_bytes;
        void *tensor_ptr;
    };

    struct DeviceAllocator
    {
        void operator()(void **ptr, size_t size) const
        {
            LOG_INFO("Malloc " << size << " bytes on device");
            TIMED(hipMalloc, check_hip_status(hipMalloc(ptr, size)));
            TIMED(hipMemset, check_hip_status(hipMemset(*ptr, 0, size)));
        }
    };

    struct DeviceFree
    {
        void operator()(void *ptr) const
        {
            TIMED(hipFree, check_hip_status_non_throwing(hipFree(ptr)));
            ptr = nullptr;
        }
    };

    struct HostAllocator
    {
        void operator()(void **ptr, size_t size) const
        {
            LOG_INFO("Malloc " << size << " bytes on host");
            TIMED(hipHostMalloc, check_hip_status(hipHostMalloc(ptr, size)));
            TIMED(hipMemset, check_hip_status(hipMemset(*ptr, 0, size)));
        }
    };

    struct HostFree
    {
        void operator()(void *ptr) const
        {
            TIMED(hipHostFree, check_hip_status_non_throwing(hipHostFree(ptr)));
            ptr = nullptr;
        }
    };

    using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
    using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

    template<typename T>
    struct ManagedBuffer_v2
    {

        explicit ManagedBuffer_v2(std::vector<T>&& host_data)
        {
            size_in_bytes = host_data.size() * sizeof(T);
            hbuff = std::move(host_data);
            dbuff = DeviceBuffer(size_in_bytes, 0);
        }

        void* data()
        {
            return dbuff.tensor_ptr;
        }

        void update(std::vector<T>&& host_data)
        {
            hbuff = std::move(host_data);
        }

        void upload_to_device(hipStream_t stream, size_t start_idx=0, size_t end_idx=0)
        {
            char* src_addr = reinterpret_cast<char*>(hbuff.data());
            char* dst_addr = static_cast<char*>(dbuff.tensor_ptr);
            size_t copy_size_in_bytes = size_in_bytes;

            size_t range_size_in_bytes = (end_idx - start_idx) * sizeof(T);
            if (range_size_in_bytes > 0)
            {
                size_t offset = start_idx * sizeof(T);
                src_addr += offset;
                dst_addr += offset;
                copy_size_in_bytes = range_size_in_bytes;
            }
            check_hip_status(hipMemcpyHtoDAsync(dst_addr, src_addr, copy_size_in_bytes, stream));
        }

        void download_from_device(hipStream_t stream, size_t start_idx=0, size_t end_idx=0)
        {
            char* src_addr = static_cast<char*>(dbuff.tensor_ptr);
            char* dst_addr = reinterpret_cast<char*>(hbuff.data());
            size_t copy_size_in_bytes = size_in_bytes;

            size_t range_size_in_bytes = (end_idx - start_idx) * sizeof(T);
            if (range_size_in_bytes > 0)
            {
                size_t offset = start_idx * sizeof(T);
                src_addr += offset;
                dst_addr += offset;
                copy_size_in_bytes = range_size_in_bytes;
            }
            check_hip_status(hipMemcpyDtoHAsync(dst_addr, src_addr, copy_size_in_bytes, stream));
        }

        void update_data(T data, size_t position, hipStream_t stream)
        {
            hbuff.at(position) = data;
            // TODO: don't copy over the entire buffer just the changed range
            // check_hip_status(hipMemcpy(get_device_ptr<void*>(), get_host_ptr<void*>(), dbuff.size_in_bytes, hipMemcpyKind::hipMemcpyHostToDevice));
            upload_to_device(stream, position, position + 1);
        }

        ManagedBuffer_v2() = delete;
        ManagedBuffer_v2(const ManagedBuffer_v2 &buf) = delete;
        ManagedBuffer_v2 &operator=(const ManagedBuffer_v2 &buf) = delete;

        DeviceBuffer dbuff;
        std::vector<T> hbuff;
        size_t size_in_bytes;
    };

    using LLama2InputBuffer = ManagedBuffer_v2<int64_t>;
    using LLama2OutputBuffer = ManagedBuffer_v2<float>;
    using LLama2PastKeyValueBuffer = ManagedBuffer_v2<half>;
    using ArgMaxOutputBuffer = ManagedBuffer_v2<int64_t>;
}
