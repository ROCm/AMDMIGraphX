#ifndef MIGRAPHX_GUARD_KERNELS_DEBUG_HPP
#define MIGRAPHX_GUARD_KERNELS_DEBUG_HPP

#include <hip/hip_runtime.h>

namespace migraphx {

inline __host__ __device__ void
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
