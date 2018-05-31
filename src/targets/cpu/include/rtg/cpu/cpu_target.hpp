#ifndef RTG_GUARD_RTGLIB_CPU_TARGET_HPP
#define RTG_GUARD_RTGLIB_CPU_TARGET_HPP

#include <rtg/program.hpp>

namespace rtg {
namespace cpu {

struct cpu_target
{
    std::string name() const;
    void apply(program& p) const;
};

} // namespace cpu

} // namespace rtg

#endif
