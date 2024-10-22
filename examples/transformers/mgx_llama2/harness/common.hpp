#pragma once

#include <hip/hip_runtime_api.h>

#include "logging.hpp"
#include "timer.hpp"

#include <cassert>
#include <vector>
#include <iostream>
#include <string>

#define TIMER_ON 0
#define TRACE_ON 0

#define assertm(exp, msg) assert(((void)msg, exp))

namespace mlinfer
{
    struct INoCopy
    {
        INoCopy() = default;
        virtual ~INoCopy() = default;
        INoCopy(const INoCopy &) = delete;
        INoCopy &operator=(const INoCopy &) = delete;
    };

    /* Helper function to split a string based on a delimiting character */
    inline std::vector<std::string>
    splitString(const std::string &input, const std::string &delimiter)
    {
        std::vector<std::string> result;
        size_t start = 0;
        size_t next = 0;
        while (next != std::string::npos)
        {
            next = input.find(delimiter, start);
            result.emplace_back(input, start, next - start);
            start = next + 1;
        }
        return result;
    }

#define check_hip_status(hip_call)                                                                                                                      \
    do                                                                                                                                                  \
    {                                                                                                                                                   \
        int status = (hip_call);                                                                                                                        \
        if (status != hipSuccess)                                                                                                                       \
        {                                                                                                                                               \
            throw std::runtime_error("hip error (" + std::to_string(status) + "): " + std::string(hipGetErrorString(static_cast<hipError_t>(status)))); \
        }                                                                                                                                               \
    } while (0);

#define check_hip_status_non_throwing(hip_call)                                                                                         \
    do                                                                                                                                  \
    {                                                                                                                                   \
        int status = (hip_call);                                                                                                        \
        if (status != hipSuccess)                                                                                                       \
        {                                                                                                                               \
            LOG_INFO("hip error (" + std::to_string(status) + "): " + std::string(hipGetErrorString(static_cast<hipError_t>(status)))); \
        }                                                                                                                               \
    } while (0);


#define CHECK(condition, error)              \
    do                                       \
    {                                        \
        if (!(condition))                    \
        {                                    \
            std::cerr << error << std::endl; \
        }                                    \
    } while (0);

#if TIMER_ON
#define TIMER_STARTV(s)              \
    static Timer timer##s(#s, true); \
    auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_START(s)         \
    static Timer timer##s(#s); \
    auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_STARTV(s)
#define TIMER_END(s)
#endif

#define TIMED(s, call)  \
    do                  \
    {                   \
        TIMER_START(s); \
        {               \
            call;       \
        }               \
        TIMER_END(s);   \
    } while (0);

} // namespace mlinfer

