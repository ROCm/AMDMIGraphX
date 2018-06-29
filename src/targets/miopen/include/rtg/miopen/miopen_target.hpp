#ifndef RTG_GUARD_RTGLIB_MIOPEN_TARGET_HPP
#define RTG_GUARD_RTGLIB_MIOPEN_TARGET_HPP

#include <rtg/program.hpp>

namespace rtg {
namespace miopen {

struct miopen_target
{
    std::string name() const;
    void apply(program& p) const;
    context get_context() const { return {}; }
};

} // namespace miopen

} // namespace rtg

#endif
