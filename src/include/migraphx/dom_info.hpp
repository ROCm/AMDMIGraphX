
#ifndef MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#define MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

struct dominator_info
{
    bool strictly_dominate(instruction_ref ins1, instruction_ref ins2);

    std::unordered_map<instruction_ref, instruction_ref> ins2idom;
};

dominator_info compute_dominator(program& p);
// dominator_info compute_dominator_naive(const program& p);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
