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
    void
    find_dom_tree(std::unordered_map<const instruction*, std::set<const instruction*>>& instr2_doms,
                  const instruction* p_ins,
                  std::unordered_map<const instruction*, const instruction*>& instr2_dom_tree,
                  std::unordered_map<const instruction*, const instruction*>& idom);

#ifdef MIGRAPHX_DEBUG_OPT
    void dump_doms(std::unordered_map<const instruction*, int>&, bool);
#endif
    bool is_split_point(instruction_ref ins);
    bool is_merge_point(instruction_ref ins);
    // whethere ins1 strictly dominates ins2
    bool strictly_dominates(const instruction* ins1, const instruction* ins2);
    // whether ins1 strictly post-dominates ins2.
    bool strictly_post_dominates(const instruction* ins1, const instruction* ins2);
    void propagate_splits(
        int num_of_streams,
        std::unordered_map<const instruction*, std::vector<std::vector<const instruction*>>>&
            concur_instrs,
        std::unordered_map<const instruction*, int>& instr2_points);

    program* p_program;
    // map instruction to its immediate dominator.
    std::unordered_map<const instruction*, const instruction*> instr2_idom;
    // map instruction to its immediate post dominator.
    std::unordered_map<const instruction*, const instruction*> instr2_ipdom;
};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
