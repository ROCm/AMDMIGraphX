#ifndef MIGRAPH_GUARD_INSTRUCTION_REF_HPP
#define MIGRAPH_GUARD_INSTRUCTION_REF_HPP

#include <list>
#include <functional>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {

struct instruction;
using instruction_ref = std::list<instruction>::iterator;

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
