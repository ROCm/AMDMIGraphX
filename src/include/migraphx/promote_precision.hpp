#ifndef MIGRAPHX_GUARD_MIGRAPHX_PROMOTE_PRECISION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PROMOTE_PRECISION_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

struct MIGRAPHX_EXPORT promote_precision
{
    std::string name() const { return "promote_precision"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PROMOTE_PRECISION_HPP
