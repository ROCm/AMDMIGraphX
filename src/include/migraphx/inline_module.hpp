#ifndef MIGRAPHX_GUARD_RTGLIB_INLINE_MODULE_HPP
#define MIGRAPHX_GUARD_RTGLIB_INLINE_MODULE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct inline_module
{
    std::string name() const { return "inline_module"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
