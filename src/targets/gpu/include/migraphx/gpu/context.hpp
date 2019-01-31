#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/machine_model.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_device
{
    using hip_event_ptr = MIGRAPHX_MANAGE_PTR(hipEvent_t, hipEventDestroy);

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
            auto status        = hipStreamCreateWithFlags(&result, hipStreamNonBlocking);

            if(status != hipSuccess)
                MIGRAPHX_THROW("Failed to allocate stream");
            return hip_stream_ptr{result};
        }

        hipStream_t get()
        {
            if(enabled(MIGRAPHX_DISABLE_NULL_STREAM{}))
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
            if(enabled(MIGRAPHX_DISABLE_NULL_STREAM{}))
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

    static hip_event_ptr create_event()
    {
        hipEvent_t event;
        auto status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to creat event");
        return hip_event_ptr{event};
    }

    void add_stream()
    {
        int num_of_streams = 1;
        assert(streams.size() == 0);
        if(enabled(MIGRAPHX_DISABLE_NULL_STREAM{}))
            num_of_streams = stream_info().num_of_streams();
        for(int i = 0; i < num_of_streams; ++i)
            streams.emplace_back(device_id);
    }

    stream& get_stream() { return streams.at(current_stream); }

    void set_stream(std::size_t n) { current_stream = n; }
    void create_events(int num_of_events)
    {
        for(int i = events.size(); i < num_of_events; ++i)
            events.emplace_back(create_event());
    }
    void record_event(int event)
    {
        for(int i = events.size(); i <= event; ++i)
            events.emplace_back(create_event());

        hipEventRecord(events.at(event).get(), streams.at(current_stream).get());
    }

    void wait_event(int event)
    {
        hipStreamWaitEvent(streams.at(current_stream).get(), events.at(event).get(), 0);
    }

    void stream_sync()
    {
        if(enabled(MIGRAPHX_DISABLE_NULL_STREAM{}))
        {
            int num_of_streams = streams.size();
            if(num_of_streams > 0)
            {
                for(int i = 0; i < num_of_streams; i++)
                    hipStreamSynchronize(streams.at(i).get());
            }
        }
    }

    private:
    std::size_t device_id      = 0;
    std::size_t current_stream = 0;
    std::vector<stream> streams;
    std::vector<shared<hip_event_ptr>> events;
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
    void set_stream(int n)
    {
        if(n >= 0)
            get_current_device().set_stream(n);
    }
    void create_events(int num_of_events) { get_current_device().create_events(num_of_events); }
    void record_event(int event) { get_current_device().record_event(event); }
    void wait_event(int event) { get_current_device().wait_event(event); }

    std::vector<argument> literals{};
    void finish()
    {
        get_current_device().stream_sync();
        gpu_sync();
    }

    private:
    // TODO: Make this a vector to support multiple devices
    std::shared_ptr<hip_device> current_device;
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
