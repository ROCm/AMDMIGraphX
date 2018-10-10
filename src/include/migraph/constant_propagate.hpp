#ifndef MIGRAPH_GUARD_RTGLIB_CONSTANT_PROPAGATE_HPP
#define MIGRAPH_GUARD_RTGLIB_CONSTANT_PROPAGATE_HPP

#include <string>

namespace migraph {

struct program;

struct constant_propagate
{
    std::string name() const { return "constant_propagate"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
