#include "dom_info.hpp"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool dom_info::strictly_dominates(const instruction* ins1, const instruction* ins2)
{
    if (ins1 != ins2) {
        const instruction* iter = ins2;
        while (instr2_idom.find(iter) != instr2_idom.end()) {
            if (ins1 == instr2_idom[iter])
                return true;
            iter = instr2_idom[iter];
        }
    }
    return false;
}

bool dom_info::strictly_post_dominates(const instruction* ins1, const instruction* ins2)
{
    if (ins1 != ins2) {
        const instruction* iter = ins2;
        while (instr2_ipdom.find(iter) != instr2_ipdom.end()) {
            if (ins1 == instr2_ipdom[iter])
                return true;
            iter = instr2_ipdom[iter];
        }
    }
    return false;
}
           
void dom_info::compute_dom(bool reversed)
{
    std::size_t num_of_instrs = p_program->size();
    if (num_of_instrs == 0)
        return;
    std::unordered_map<const instruction*, std::set<const instruction*>> instr2_doms;
    std::unordered_map<const instruction*, int> instr2_points;
    int cur_points = reversed ? num_of_instrs - 1 : 0;
    bool has_stream = false;
    program_visitor vis{p_program, reversed};
    std::unordered_map<const instruction*, const instruction*>& instr2_dom_tree = (reversed ? instr2_ipdom : instr2_idom);
    for (auto ins = vis.begin(), end = vis.end(); ; ins = vis.next(ins))
    {        
        const instruction* p_ins = &(*ins);
        instr2_points[p_ins] = cur_points;
        if (ins->get_stream() < 0) {
            if (reversed)
                cur_points--;
            else
                cur_points++;;
            if (ins == end)
                break;
            continue;
        }
        has_stream = true;
        const instruction* p_tmp = nullptr;
        int cnt = 0;
        // find dominators.
        for(auto& p_arg : vis.get_inputs(ins)) {
            if (p_arg->get_stream() < 0)
                continue;
            cnt++;
            assert(instr2_doms.find(p_arg) != instr2_doms.end());
            if (p_tmp == nullptr)
                instr2_doms[p_ins] = instr2_doms[p_arg];
            else 
                instr2_doms[p_ins] = set_op::set_intersection(instr2_doms[p_ins], instr2_doms[p_arg]);
            p_tmp = p_arg;
        }
        // find immediate dominators.
        if (cnt == 1) {
            instr2_dom_tree[p_ins] = p_tmp;
        } else if (cnt > 0) {
            for (auto& iter1 : instr2_doms[p_ins]) {
                bool is_idom = true;
                // check whether iter1 strictly dominates or post-dominates any other notes in p_ins's dominators or post-dominators.
                for (auto& iter2 : instr2_doms[p_ins]) {
                    if ((reversed && strictly_post_dominates(iter1, iter2))
                        || (!reversed && strictly_dominates(iter1, iter2))) {
                        is_idom = false;
                        break;
                    }
                }
                if (is_idom) {
                    assert(instr2_dom_tree.find(p_ins) == instr2_dom_tree.end());
                    instr2_dom_tree[p_ins] = iter1;
                }
            }
        }
        
        instr2_doms[p_ins].insert(p_ins);
        if (ins == end)
            break;
        if (reversed)
            cur_points--;
        else
            cur_points++;
    }
    if (has_stream) {
        MIGRAPHX_DEBUG(dump_doms(instr2_points, reversed));
    }
}

#ifdef MIGRAPHX_DEBUG_OPT

void dom_info::dump_doms(std::unordered_map<const instruction*, int>& instr2_points, bool post_dom)
{
    std::cout << "---dominator tree---" << std::endl;
    for(auto ins : iterator_for(*p_program))
    {
        const instruction* p_ins = &(*ins);
        if (!post_dom && (instr2_idom.find(p_ins) != instr2_idom.end())) {
            const instruction* idom = instr2_idom[p_ins];
            std::cout << "@" << instr2_points[p_ins] << " imm dominator: " << "@" << instr2_points[idom] << std::endl;
        }
        if (post_dom && (instr2_ipdom.find(p_ins) != instr2_ipdom.end())) {
            const instruction* ipdom = instr2_ipdom[p_ins];
            std::cout << "@" << instr2_points[p_ins] << " imm post domimator: " << "@" << instr2_points[ipdom] << std::endl;

        }
    }
}
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
