#ifndef MIGRAPHX_GUARD_RTGLIB_NAME_HPP
#define MIGRAPHX_GUARD_RTGLIB_NAME_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// Create name from class
template <class Derived>
struct op_name
{
    std::string name() const
    {
        static const std::string& name = get_type_name<Derived>();
        return name.substr(name.rfind("::") + 2);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
