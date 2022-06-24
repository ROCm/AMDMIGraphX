/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/context.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>
#include <unordered_map>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_NULL_STREAM)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_NSTREAMS)

using hip_event_ptr = MIGRAPHX_MANAGE_PTR(hipEvent_t, hipEventDestroy);

struct hip_device
{
    hip_device()
    {
        device_props.gcnArchName[0]      = '\0';
        device_props.gcnArch             = 0;
        device_props.multiProcessorCount = 0;
        add_stream();
    }

    hip_device(std::size_t id, std::size_t n) : device_id(id)
    {
        auto status = hipGetDeviceProperties(&device_props, device_id);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to allocate stream");

        for(std::size_t i = 0; i < n; i++)
            add_stream();
    }

    struct stream
    {
        using hip_stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

        stream() {}

        stream(std::size_t device_number) : id(device_number) {}

        void setup() const { set_device(id); }

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

        void wait() const
        {
            if(s == nullptr)
                return;
            setup();
            auto status = hipStreamSynchronize(s.get());
            if(status != hipSuccess)
                MIGRAPHX_THROW("Failed to wait.");
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

    const stream& get_stream() const { return streams.at(current_stream); }

    const stream& get_stream(std::size_t n) const { return streams.at(n); }

    void set_stream(std::size_t n) { current_stream = n; }

    std::size_t nstreams() const { return streams.size(); }

    std::size_t stream_id() const { return current_stream; }

    std::string get_device_name() const { return device_props.gcnArchName; }

    std::size_t get_device_major() const { return device_props.major; }

    std::size_t get_device_minor() const { return device_props.minor; }

    std::size_t get_cu_count() const { return device_props.multiProcessorCount; }

    std::size_t get_max_workitems_per_cu() const
    {
        return device_props.maxThreadsPerMultiProcessor;
    }

    std::size_t get_max_workitems_per_block() const { return device_props.maxThreadsPerBlock; }

    private:
    std::size_t device_id      = 0;
    std::size_t current_stream = 0;
    std::vector<stream> streams;
    hipDeviceProp_t device_props;

    public:
    std::unordered_map<std::string, argument> preallocations{};
};

struct context
{
    context(std::size_t device_id = 0, std::size_t n = value_of(MIGRAPHX_NSTREAMS{}, 1))
        : current_device(std::make_shared<hip_device>(device_id, n))
    {
    }

    hip_device& get_current_device()
    {
        assert(current_device != nullptr);
        return *current_device;
    }

    const hip_device& get_current_device() const
    {
        assert(current_device != nullptr);
        return *current_device;
    }

    hip_device::stream& get_stream() { return get_current_device().get_stream(); }
    hip_device::stream& get_stream(std::size_t n) { return get_current_device().get_stream(n); }

    const hip_device::stream& get_stream() const { return get_current_device().get_stream(); }
    const hip_device::stream& get_stream(std::size_t n) const
    {
        return get_current_device().get_stream(n);
    }

    void set_stream(std::size_t n) { get_current_device().set_stream(n); }

    void create_events(std::size_t num_of_events)
    {
        for(std::size_t i = events.size(); i < num_of_events + 1; ++i)
            events.emplace_back(create_event());
    }

    hipEvent_t get_event(std::size_t i) const { return events.at(i).get(); }

    std::vector<argument> literals{};
    void finish() const { get_stream().wait(); }

    static hip_event_ptr create_event()
    {
        hipEvent_t event;
        auto status = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to create event");
        return hip_event_ptr{event};
    }

    value to_value() const
    {
        value result;
        result["events"]  = events.size();
        result["streams"] = current_device->nstreams();

        return result;
    }

    void from_value(const value& v)
    {
        auto v_events        = v.at("events");
        std::size_t n_events = v_events.without_key().to<std::size_t>();
        this->create_events(n_events - 1);

        auto v_streams        = v.at("streams");
        std::size_t n_streams = v_streams.without_key().to<std::size_t>();

        this->current_device = std::make_shared<hip_device>(0, n_streams);
    }

    any_ptr get_queue() { return get_stream().get(); }

    private:
    // TODO: Make this a vector to support multiple devices
    std::shared_ptr<hip_device> current_device;
    std::vector<shared<hip_event_ptr>> events;
};

inline void migraphx_to_value(value& v, const context& ctx) { v = ctx.to_value(); }
inline void migraphx_from_value(const value& v, context& ctx) { ctx.from_value(v); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
