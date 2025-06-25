#ifndef MIGRAPHX_GUARD_MIGRAPHX_RETURNS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_RETURNS_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Similiar to decltype(auto) except it will propagate any substitution failures
// NOLINTNEXTLINE
#define MIGRAPHX_RETURNS(...) \
    ->decltype(__VA_ARGS__) { return __VA_ARGS__; }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_RETURNS_HPP
