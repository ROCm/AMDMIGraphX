#ifndef MIGRAPHX_GUARD_MIGRAPHX_LIFETIME_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LIFETIME_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

enum class lifetime
{
    local,
    global,
    borrow
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_LIFETIME_HPP
