#ifndef MIGRAPHX_GUARD_MODULE_REF_HPP
#define MIGRAPHX_GUARD_MODULE_REF_HPP

#include <list>
#include <functional>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
using module_ref = module*;
using const_module_ref = const module*;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
