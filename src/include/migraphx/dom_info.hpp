#ifndef MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#define MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#include <migraphx/common_header.hpp>
#include <migraphx/set_operator.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct dom_info
{
    dom_info(program* p) : p_program(p)
    {
        instr2_idom.clear();
        instr2_ipdom.clear();
    }
    void compute_dom(bool);

#ifdef MIGRAPHX_DEBUG_OPT
    void dump_doms(std::unordered_map<const instruction*, int>&, bool);
#endif
    // whethere ins1 strictly dominates ins2
    bool strictly_dominates(const instruction* ins1, const instruction* ins2);
    // whether ins1 strictly post-dominates ins2.
    bool strictly_post_dominates(const instruction* ins1, const instruction* ins2);
    static instruction* get_stream(program* p, instruction_ref ins);
    program* p_program;
    // map instruction to its immediate dominator.
    std::unordered_map<const instruction*, const instruction*> instr2_idom;
    // map instruction to its immediate post dominator.
    std::unordered_map<const instruction*, const instruction*> instr2_ipdom;
};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
