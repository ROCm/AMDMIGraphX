#ifndef MIGRAPHX_GUARD_RTGLIB_NAME_HPP
#define MIGRAPHX_GUARD_RTGLIB_NAME_HPP

#include <migraphx/config.hpp>
#include <migraphx/type_name.hpp>

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
