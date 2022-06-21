#ifndef MIGRAPHX_GUARD_MIGRAPHX_PREALLOCATE_PARAM_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PREALLOCATE_PARAM_HPP

#include <migraphx/config.hpp>
#include <migraphx/allocation_model.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct preallocate_param
{
    std::string param;
    allocation_model model;
    std::string name() const { return "preallocate_param"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PREALLOCATE_PARAM_HPP
