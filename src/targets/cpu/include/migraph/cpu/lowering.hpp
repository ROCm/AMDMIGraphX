#ifndef MIGRAPH_GUARD_RTGLIB_CPU_LOWERING_HPP
#define MIGRAPH_GUARD_RTGLIB_CPU_LOWERING_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace cpu {

struct lowering
{
    std::string name() const { return "cpu::lowering"; }
    void apply(program& p) const;
};

} // namespace cpu

} // namespace migraph

#endif
