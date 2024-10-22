#pragma once

#include <iostream>

namespace mlinfer
{

#define LOGGING_OFF 0
#define ENABLE_TIMED_LOGGING 0
#define ENABLE_DEBUG_LOGGING 0

#if (!LOGGING_OFF)
#define LOG_INFO(...)                          \
    do                                         \
    {                                          \
        std::cout << __VA_ARGS__ << std::endl; \
    } while (0)
#define LOG_ERROR(...)                         \
    do                                         \
    {                                          \
        std::cerr << __VA_ARGS__ << std::endl; \
    } while (0)
#define LOG_STATE(...)                                                                \
    do                                                                                \
    {                                                                                 \
        std::cout << "================================================" << std::endl; \
        std::cout << __VA_ARGS__ << std::endl;                                        \
        std::cout << "================================================" << std::endl; \
    } while (0)
#else
#define LOG_INFO(...) (void)0
#define LOG_ERROR(...) (void)0
#define LOG_STATE(...) (void)0
#endif

#if (ENABLE_TIMED_LOGGING || ENABLE_DEBUG_LOGGING)
#define LOG_TIMED(...) LOG_INFO(__VA_ARGS__)
#else
#define LOG_TIMED(...) (void)0
#endif

#if ENABLE_DEBUG_LOGGING
#define LOG_DEBUG(...) LOG_INFO(__VA_ARGS__)
#else
#define LOG_DEBUG(...) (void)0
#endif

} // namespace mlinfer

