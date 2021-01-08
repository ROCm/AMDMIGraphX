#ifndef MIGRAPHX_GUARD_RTGLIB_AUTO_CONTIGOUS_HPP
#define MIGRAPHX_GUARD_RTGLIB_AUTO_CONTIGOUS_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct auto_contiguous
{
    std::string name() const { return "auto_contiguous"; }
    void apply(module& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
