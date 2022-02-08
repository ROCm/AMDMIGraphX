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
    const auto size = (sizeof(xs) + ...);
    print_buffer<size> buffer;
    swallow{(buffer.append(xs), 0)...};
    printf("%s", buffer.buffer);
}

} // namespace debug

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

#ifdef MIGRAPHX_DEBUG
#define MIGRAPHX_ASSERT(cond)                               \
    ((cond) ? void(0) : [](auto&&... private_migraphx_xs) { \
        assert_fail(private_migraphx_xs...);                \
    }(#cond, __FILE__, MIGRAPHX_STRINGIZE(__LINE__), __PRETTY_FUNCTION__))
#else
#define MIGRAPHX_ASSERT(cond)
#endif

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_DEBUG_HPP
