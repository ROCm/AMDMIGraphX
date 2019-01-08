#ifndef MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#define MIGRAPHX_GUARD_RTGLIB_DOM_INFO_HPP
#include "common_header.hpp"
#include "set_operator.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program_visitor
{
    program* p_program;
    bool reversed;
    instruction_ref begin()
    { return reversed ? std::prev(p_program->end()) : p_program->begin(); }
    instruction_ref end()
    {
        return reversed ? p_program->begin() : std::prev(p_program->end());
    }
    
    instruction_ref next(instruction_ref ins)
    {
        return reversed ? std::prev(ins) : std::next(ins);
    }
    std::set<const instruction*> get_inputs(instruction_ref ins)
    {
        std::set<const instruction*> ret_val;
        for (auto&& arg : reversed ? ins->outputs() : ins->inputs())
            ret_val.insert(&(*arg));
        return ret_val;
    }
};

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
    program* p_program;
    // map instruction to its immediate dominator.
    std::unordered_map<const instruction*, const instruction*> instr2_idom;
    // map instruction to its immediate post dominator.
    std::unordered_map<const instruction*, const instruction*> instr2_ipdom;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
