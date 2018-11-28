#ifndef MIGRAPHX_GUARD_RTGLIB_CPU_LOWERING_HPP
#define MIGRAPHX_GUARD_RTGLIB_CPU_LOWERING_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct lowering
{
    std::string name() const { return "cpu::lowering"; }
    void apply(program& p) const;
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
