#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_DOT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_DOT_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct MIGRAPHX_EXPORT rewrite_dot
{
    std::string name() const { return "rewrite_dot"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_DOT_HPP

