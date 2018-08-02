#ifndef MIGRAPH_GUARD_RTGLIB_AUTO_CONTIGOUS_HPP
#define MIGRAPH_GUARD_RTGLIB_AUTO_CONTIGOUS_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct auto_contigous
{
    std::string name() const { return "auto_contigous"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
