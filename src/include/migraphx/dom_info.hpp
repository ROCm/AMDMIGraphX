
#ifndef MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#define MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct dominator_info
{
    bool strictly_dominate(instruction_ref ins1, instruction_ref ins2);

    std::unordered_map<instruction_ref, instruction_ref> ins2idom;
};

dominator_info compute_dominator(module& m);
// dominator_info compute_dominator_naive(const module& m);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
