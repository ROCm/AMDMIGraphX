#ifndef RTG_GUARD_INSTRUCTION_REF_HPP
#define RTG_GUARD_INSTRUCTION_REF_HPP

#include <list>

namespace rtg {

struct instruction;
using instruction_ref = std::list<instruction>::iterator;

} // namespace rtg

#endif
