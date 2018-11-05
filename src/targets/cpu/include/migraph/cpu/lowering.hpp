#ifndef MIGRAPH_GUARD_RTGLIB_CPU_LOWERING_HPP
#define MIGRAPH_GUARD_RTGLIB_CPU_LOWERING_HPP

#include <migraph/program.hpp>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

struct lowering
{
    std::string name() const { return "cpu::lowering"; }
    void apply(program& p) const;
};

} // namespace cpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
