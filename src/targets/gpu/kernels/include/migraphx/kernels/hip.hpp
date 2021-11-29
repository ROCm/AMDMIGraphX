#ifndef MIGRAPHX_GUARD_KERNELS_HIP_HPP
#define MIGRAPHX_GUARD_KERNELS_HIP_HPP

// Workaround macro redefinition issue with clang tidy
#if defined(__HIP_PLATFORM_HCC__) && defined(MIGRAPHX_USE_CLANG_TIDY)
#undef __HIP_PLATFORM_HCC__ // NOLINT
#endif

#include <hip/hip_runtime.h>

#endif // MIGRAPHX_GUARD_KERNELS_HIP_HPP
