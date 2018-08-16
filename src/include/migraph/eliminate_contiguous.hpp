#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONTIGUOUS_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONTIGUOUS_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct eliminate_contiguous
{
    std::string name() const { return "eliminate_contiguous"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
