#ifndef MIGRAPH_GUARD_INSTRUCTION_REF_HPP
#define MIGRAPH_GUARD_INSTRUCTION_REF_HPP

#include <list>
#include <functional>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct instruction;
using instruction_ref = std::list<instruction>::iterator;

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
