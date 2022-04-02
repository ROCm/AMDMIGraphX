#ifndef MIGRAPHX_GUARD_KERNELS_DEBUG_HPP
#define MIGRAPHX_GUARD_KERNELS_DEBUG_HPP

#include <migraphx/kernels/hip.hpp>

namespace migraphx {

#define MIGRAPHX_STRINGIZE_1(...) #__VA_ARGS__
#define MIGRAPHX_STRINGIZE(...) MIGRAPHX_STRINGIZE_1(__VA_ARGS__)

// Workaround hip's broken abort on device code
#ifdef __HIP_DEVICE_COMPILE__
// NOLINTNEXTLINE
#define MIGRAPHX_HIP_NORETURN
#else
// NOLINTNEXTLINE
#define MIGRAPHX_HIP_NORETURN [[noreturn]]
#endif

namespace debug {
struct swallow
{
    template <class... Ts>
    constexpr swallow(Ts&&...)
    {
    }
};

template <size_t N>
struct print_buffer
{
    char buffer[N + 1] = {0};
    char* pos          = buffer;

    constexpr void append(char c)
    {
        if(c == 0)
            return;
        if(pos < buffer + N)
        {
            *pos = c;
            pos++;
        }
    }
    constexpr void append(int i)
    {
        if (i < 0)
        {
            append('-');
            i = -i;
        }
        while(i > 0)
        {
            char c = (i%10) + '0';
            append(c);
            i /= 10;
        }
    }

    template <size_t M>
    constexpr void append(const char (&array)[M])
    {
        for(int i = 0; i < M; i++)
            append(array[i]);
    }
};

template <class... Ts>
__host__ __device__ void print(const Ts&... xs)
{
    print_buffer<1024> buffer;
    swallow{(buffer.append(xs), 0)...};
    printf("%s", buffer.buffer);
}

} // namespace debug

struct source_location
{
    int line = __builtin_LINE();
    const char* file = __builtin_FILE();
    const char* function = __builtin_FUNCTION();
};

// noreturn cannot be used on this function because abort in hip is broken
template <class T1, class T2, class T3, class T4>
MIGRAPHX_HIP_NORETURN inline __host__ __device__ void
assert_fail(const T1& assertion, const T2& file, const T3& line, const T4& function)
{
    // printf is broken on hip with more than one argument, so use a simple print functions instead
    debug::print(file, ":", line, ": ", function, ": assertion '", assertion, "' failed.\n");
    // printf("%s:%s: %s: assertion '%s' failed.\n", file, line, function, assertion);
    abort();
}

template <class T1, class T2, class T3, class T4>
MIGRAPHX_HIP_NORETURN inline __host__ __device__ void
assert_fail(const T1& message, const source_location& loc)
{
    debug::print(loc.file, ":", loc.line, ": ", loc.function, ": ", message, "\n");
    abort();
}

// NOLINTNEXTLINE
#define MIGRAPHX_ASSERT_FAIL(cond, ...)                                \
    ((cond) ? void(0) : [](auto&&... private_migraphx_xs) { \
        assert_fail(private_migraphx_xs...);                \
    }(__VA_ARGS__))

// NOLINTNEXTLINE
#define MIGRAPHX_CHECK(cond) MIGRAPHX_ASSERT_FAIL(cond, #cond, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#ifdef MIGRAPHX_DEBUG
#define MIGRAPHX_CAPTURE_SOURCE_LOCATION(loc) , source_location loc = source_location{}
#define MIGRAPHX_ASSERT2(cond, msg, loc) MIGRAPHX_ASSERT_FAIL(cond, msg, loc)
#define MIGRAPHX_ASSERT MIGRAPHX_CHECK
#define MIGRAPHX_ASSUME MIGRAPHX_CHECK
#define MIGRAPHX_UNREACHABLE() MIGRAPHX_ASSERT(false)
#else
#define MIGRAPHX_CAPTURE_SOURCE_LOCATION(loc)
#define MIGRAPHX_ASSUME __builtin_assume
#define MIGRAPHX_UNREACHABLE __builtin_unreachable
#define MIGRAPHX_ASSERT(cond)
#define MIGRAPHX_ASSERT2(...)
#endif

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_DEBUG_HPP
