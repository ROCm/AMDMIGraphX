#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONCAT_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONCAT_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/concat_opt.hpp>

namespace migraph {

struct program;

struct eliminate_concat
{
	concat_optimization concat_opt;
    std::string name() const { return "eliminate_concat"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
