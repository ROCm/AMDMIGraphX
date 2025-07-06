#ifndef MIGRAPHX_GUARD_MIGRAPHX_PMR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PMR_HPP

#include <migraphx/config.hpp>

#if defined(__has_include) && __has_include(<memory_resource>)
#include <memory_resource>
#endif

#if defined(__cpp_lib_memory_resource) && __cpp_lib_memory_resource >= 201603L
#define MIGRAPHX_HAS_PMR 1
#endif

#ifndef MIGRAPHX_HAS_PMR
#define MIGRAPHX_HAS_PMR 0
#endif

#endif // MIGRAPHX_GUARD_MIGRAPHX_PMR_HPP
