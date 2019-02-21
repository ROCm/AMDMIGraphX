#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_NULL_STREAM)

struct hip_device
{
    hip_device() { add_stream(); }

    hip_device(std::size_t id) : device_id(id) { add_stream(); }

    struct stream
    {
        using hip_stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

        stream() {}

        stream(std::size_t device_number) : id(device_number) {}

        void setup() { set_device(id); }

        static hip_stream_ptr create_stream()
        {
            hipStream_t result = nullptr;
            auto status        = hipStreamCreate(&result);
            if(status != hipSuccess)
                MIGRAPHX_THROW("Failed to allocate stream");
            return hip_stream_ptr{result};
        }

        hipStream_t get()
        {
            if(not enabled(MIGRAPHX_ENABLE_NULL_STREAM{}))
            {
                setup();
                if(s == nullptr)
                    s = create_stream();
                assert(s.get() != nullptr);
                return s.get();
            }
            return nullptr;
        }

        auto create_miopen_handle()
        {
            if(not enabled(MIGRAPHX_ENABLE_NULL_STREAM{}))
                return make_obj<miopen_handle>(&miopenCreateWithStream, get());
            else
                return make_obj<miopen_handle>(&miopenCreate);
        }

        auto get_miopen()
        {
            setup();
            if(mihandle == nullptr)
                mihandle = create_miopen_handle();
            assert(mihandle.get() != nullptr);
            return mihandle.get();
        }

        auto get_rocblas()
        {
            setup();
            if(rbhandle == nullptr)
                rbhandle = create_rocblas_handle_ptr(get());
            assert(rbhandle.get() != nullptr);
            return rbhandle.get();
        }

        private:
        std::size_t id                      = 0;
        shared<hip_stream_ptr> s            = nullptr;
        shared<miopen_handle> mihandle      = nullptr;
        shared<rocblas_handle_ptr> rbhandle = nullptr;
    };

    void add_stream() { streams.emplace_back(device_id); }

    stream& get_stream() { return streams.at(current_stream); }

    void set_stream(std::size_t n) { current_stream = n; }

    private:
    std::size_t device_id      = 0;
    std::size_t current_stream = 0;
    std::vector<stream> streams;
};

struct context
{
    context(std::size_t n = 0) : current_device(std::make_shared<hip_device>(n)) {}

    hip_device& get_current_device()
    {
        assert(current_device != nullptr);
        return *current_device;
    }

    hip_device::stream& get_stream() { return get_current_device().get_stream(); }

    std::vector<argument> literals{};
    void finish() const { gpu_sync(); }

    private:
    // TODO: Make this a vector to support multiple devices
    std::shared_ptr<hip_device> current_device;
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
