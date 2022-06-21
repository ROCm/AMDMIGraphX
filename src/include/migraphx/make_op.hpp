#ifndef MIGRAPHX_GUARD_RTGLIB_MAKE_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_MAKE_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

operation make_op(const std::string& name);
operation make_op(const std::string& name,
                  const std::initializer_list<std::pair<std::string, value>>& v);
operation make_op_from_value(const std::string& name, const value& v);

// A template overload is added for migraphx::value so the initializer_list
// cannot be passed in directly. This is to enforce at compile-time that all
// initializer_list are key-value pairs, whereas migraphx::value allows other
// types of initializer_list such as for arrays.
template <class Value>
operation make_op(const std::string& name, const Value& v)
{
    return make_op_from_value(name, v);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
