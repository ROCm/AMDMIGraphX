#ifndef MIGRAPH_GUARD_INSTRUCTION_REF_HPP
#define MIGRAPH_GUARD_INSTRUCTION_REF_HPP

#include <list>
#include <functional>

namespace migraph {

struct instruction;
using instruction_ref = std::list<instruction>::iterator;

} // namespace migraph

#endif
