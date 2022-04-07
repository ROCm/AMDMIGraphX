#ifndef MIGRAPHX_GUARD_RTGLIB_CPU_LOWERING_HPP
#define MIGRAPHX_GUARD_RTGLIB_CPU_LOWERING_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace cpu {

struct lowering
{
    std::string name() const { return "cpu::lowering"; }
    void apply(module& m) const;
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
