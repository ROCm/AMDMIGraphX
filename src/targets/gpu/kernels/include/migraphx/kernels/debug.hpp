#ifndef MIGRAPHX_GUARD_KERNELS_DEBUG_HPP
#define MIGRAPHX_GUARD_KERNELS_DEBUG_HPP

#include <migraphx/kernels/hip.hpp>

namespace migraphx {

// Workaround hip's broken abort on device code
#ifdef __HIP_DEVICE_COMPILE__
// NOLINTNEXTLINE
#define MIGRAPHX_HIP_NORETURN
#else
// NOLINTNEXTLINE
#define MIGRAPHX_HIP_NORETURN [[noreturn]]
#endif

// noreturn cannot be used on this function because abort in hip is broken
MIGRAPHX_HIP_NORETURN inline __host__ __device__ void
assert_fail(const char* assertion, const char* file, unsigned int line, const char* function)
{
    printf("%s:%u: %s: assertion '%s' failed.\n", file, line, function, assertion);
    abort();
}

#ifdef MIGRAPHX_DEBUG
#define MIGRAPHX_ASSERT(cond)            \
    ((cond) ? void(0) : [](auto... xs) { \
        assert_fail(xs...);              \
    }(#cond, __FILE__, __LINE__, __PRETTY_FUNCTION__))
#else
#define MIGRAPHX_ASSERT(cond)
#endif

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_DEBUG_HPP
