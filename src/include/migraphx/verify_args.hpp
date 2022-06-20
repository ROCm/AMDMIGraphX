#ifndef MIGRAPHX_GUARD_RTGLIB_VERIFY_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_VERIFY_ARGS_HPP

#include <migraphx/verify.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool verify_args(const std::string& name,
                 const argument& ref_arg,
                 const argument& target_arg,
                 double tolerance = 80);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
