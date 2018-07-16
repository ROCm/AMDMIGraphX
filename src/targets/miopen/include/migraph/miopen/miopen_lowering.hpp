#ifndef MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_HPP
#define MIGRAPH_GUARD_RTGLIB_MIOPEN_LOWERING_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace miopen {

struct miopen_lowering
{
    std::string name() const { return "miopen::lowering"; }
    void apply(program& p) const;
};

} // namespace miopen

} // namespace migraph

#endif
