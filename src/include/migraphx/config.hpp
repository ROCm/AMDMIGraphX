#ifndef MIGRAPHX_GUARD_CONFIG_HPP
#define MIGRAPHX_GUARD_CONFIG_HPP

namespace migraphx {

#if !defined(MIGRAPHX_USE_CLANG_TIDY) && !defined(DOXYGEN)
#define MIGRAPHX_INLINE_NS version_1
#endif

#ifdef DOXYGEN
#define MIGRAPHX_INLINE_NS internal
#endif

#ifdef MIGRAPHX_USE_CLANG_TIDY
#define MIGRAPHX_TIDY_CONST const
#else
#define MIGRAPHX_TIDY_CONST
#endif

} // namespace migraphx

#endif
