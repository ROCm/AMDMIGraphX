#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONTIGUOUS_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONTIGUOUS_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {

struct program;

struct eliminate_contiguous
{
    std::string name() const { return "eliminate_contiguous"; }
    void apply(program& p) const;
};

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
