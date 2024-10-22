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

    struct ManagedBuffer
    {

        explicit ManagedBuffer(size_t size_in_bytes_, size_t stride_in_bytes_ = 0)
        {
            dbuff = DeviceBuffer(size_in_bytes_, stride_in_bytes_);
            hbuff = HostBuffer(size_in_bytes_, stride_in_bytes_);
        }

        template <typename T>
        T get_host_ptr()
        {
            return static_cast<T>(hbuff.tensor_ptr);
        }

        template <typename T>
        T get_device_ptr()
        {
            return static_cast<T>(dbuff.tensor_ptr);
        }

        void upload_to_device(void* data, size_t size_in_bytes)
        {
            memcpy(get_host_ptr<void*>(), data, size_in_bytes);
            check_hip_status(hipMemcpy(get_device_ptr<void*>(), get_host_ptr<void*>(), size_in_bytes, hipMemcpyKind::hipMemcpyHostToDevice));
        }

        template <typename T>
        std::vector<T> download_from_device(size_t size_in_bytes)
        {
            check_hip_status(hipMemcpy(get_host_ptr<void*>(), get_device_ptr<void*>(), size_in_bytes, hipMemcpyKind::hipMemcpyDeviceToHost));
            return std::vector<T>(get_host_ptr<T*>(), get_host_ptr<T*>() + (size_in_bytes / sizeof(T)));
        }

        template <typename T>
        void update_device_data(T data, size_t position)
        {
            T* host_data = get_host_ptr<T*>();
            host_data[position] = data;
            // TODO: don't copy over the entire buffer just the changed range
            check_hip_status(hipMemcpy(get_device_ptr<void*>(), get_host_ptr<void*>(), dbuff.size_in_bytes, hipMemcpyKind::hipMemcpyHostToDevice));
        }

        ManagedBuffer() = delete;
        ManagedBuffer(const ManagedBuffer &buf) = delete;
        ManagedBuffer &operator=(const ManagedBuffer &buf) = delete;

        DeviceBuffer dbuff;
        HostBuffer hbuff;
    };
}
