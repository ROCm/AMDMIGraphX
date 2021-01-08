#ifndef MIGRAPHX_GUARD_RTGLIB_ADJUST_ALLOCATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_ADJUST_ALLOCATION_HPP

#include <migraphx/config.hpp>
#include <migraphx/allocation_model.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct adjust_allocation
{
    allocation_model model;
    std::string name() const { return "adjust_allocation"; }
    void apply(module& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
