#ifndef MIGRAPHX_GUARD_INSTRUCTION_REF_HPP
#define MIGRAPHX_GUARD_INSTRUCTION_REF_HPP

#include <list>
#include <functional>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct instruction;
using instruction_ref = std::list<instruction>::iterator;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
