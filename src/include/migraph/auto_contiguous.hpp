#ifndef MIGRAPH_GUARD_RTGLIB_AUTO_CONTIGOUS_HPP
#define MIGRAPH_GUARD_RTGLIB_AUTO_CONTIGOUS_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct auto_contiguous
{
    std::string name() const { return "auto_contiguous"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
