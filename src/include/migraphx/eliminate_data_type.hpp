#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ELIMINATE_DATA_TYPE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ELIMINATE_DATA_TYPE_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <set>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Remove data types. This will instert convert operators so the data type
 * is not used by any operator.
 */
struct eliminate_data_type
{
    std::set<shape::type_t> types;
    shape::type_t target_type;
    std::string name() const { return "eliminate_data_type"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
