#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_NULL_STREAM)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_NSTREAMS)

using hip_event_ptr = MIGRAPHX_MANAGE_PTR(hipEvent_t, hipEventDestroy);

struct hip_device
{
    hip_device() { add_stream(); }

    hip_device(std::size_t id, std::size_t n) : device_id(id)
    {
        for(std::size_t i = 0; i < n; i++)
            add_stream();
    }

    struct stream
    {
        using hip_stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

        stream() {}

        stream(std::size_t device_number) : id(device_number) {}

        void setup() { set_device(id); }

        static hip_stream_ptr create_stream()
        {
            hipStream_t result = nullptr;
            auto status        = hipStreamCreateWithFlags(&result, hipStreamNonBlocking);
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

        void wait(hipEvent_t event)
        {
            setup();
            auto status = hipStreamWaitEvent(get(), event, 0);
            if(status != hipSuccess)
                MIGRAPHX_THROW("Failed to wait.");
        }

        void record(hipEvent_t event)
        {
            setup();
            auto status = hipEventRecord(event, get());
            if(status != hipSuccess)
                MIGRAPHX_THROW("Failed to record.");
        }

        private:
        std::size_t id                      = 0;
        shared<hip_stream_ptr> s            = nullptr;
        shared<miopen_handle> mihandle      = nullptr;
        shared<rocblas_handle_ptr> rbhandle = nullptr;
    };

    void add_stream() { streams.emplace_back(device_id); }

    stream& get_stream() { return streams.at(current_stream); }

    stream& get_stream(std::size_t n) { return streams.at(n); }

    void set_stream(std::size_t n) { current_stream = n; }

    std::size_t nstreams() const { return streams.size(); }

    std::size_t stream_id() const { return current_stream; }

    private:
    std::size_t device_id      = 0;
    std::size_t current_stream = 0;
    std::vector<stream> streams;

    public:
    std::unordered_map<std::string, argument> preallocations{};
};

struct context
{
    context(std::size_t device_id = 0, std::size_t n = value_of(MIGRAPHX_NSTREAMS{}, 4))
        : current_device(std::make_shared<hip_device>(device_id, n))
    {
    }

    hip_device& get_current_device()
    {
        assert(current_device != nullptr);
        return *current_device;
    }

    hip_device::stream& get_stream() { return get_current_device().get_stream(); }
    hip_device::stream& get_stream(std::size_t n) { return get_current_device().get_stream(n); }

    void set_stream(std::size_t n) { get_current_device().set_stream(n); }

    void create_events(std::size_t num_of_events)
    {
        for(std::size_t i = events.size(); i < num_of_events + 1; ++i)
            events.emplace_back(create_event());
    }

    hipEvent_t get_event(std::size_t i) const { return events.at(i).get(); }

    std::vector<argument> literals{};
    void finish() const { gpu_sync(); }

    static hip_event_ptr create_event()
    {
        hipEvent_t event;
        auto status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to create event");
        return hip_event_ptr{event};
    }

    private:
    // TODO: Make this a vector to support multiple devices
    std::shared_ptr<hip_device> current_device;
    std::vector<shared<hip_event_ptr>> events;
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
