#ifndef MIGRAPHX_GUARD_RTGLIB_MAKE_OP_HPP
#define MIGRAPHX_GUARD_RTGLIB_MAKE_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

operation make_op(const std::string& name);
operation make_op(const std::string& name, const value& v);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
