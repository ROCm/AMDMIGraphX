#include <migraphx/dom_info.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program_visitor
{
    program* p_program;
    bool reversed;
    instruction_ref begin() { return reversed ? std::prev(p_program->end()) : p_program->begin(); }
    instruction_ref end() { return reversed ? p_program->begin() : std::prev(p_program->end()); }

    instruction_ref next(instruction_ref ins) { return reversed ? std::prev(ins) : std::next(ins); }
    const std::vector<instruction_ref>& get_inputs(instruction_ref ins)
    {
        return reversed ? ins->outputs() : ins->inputs();
    }
};

bool dom_info::strictly_dominates(const instruction* ins1, const instruction* ins2)
{
    if(ins1 != ins2)
    {
        const instruction* iter = ins2;
        while(instr2_idom.find(iter) != instr2_idom.end())
        {
            if(ins1 == instr2_idom[iter])
                return true;
            iter = instr2_idom[iter];
        }
    }
    return false;
}

bool dom_info::strictly_post_dominates(const instruction* ins1, const instruction* ins2)
{
    if(ins1 != ins2)
    {
        const instruction* iter = ins2;
        while(instr2_ipdom.find(iter) != instr2_ipdom.end())
        {
            if(ins1 == instr2_ipdom[iter])
                return true;
            iter = instr2_ipdom[iter];
        }
    }
    return false;
}

instruction * dom_info::get_stream(program * p, instruction_ref ins)
{
    instruction_ref iter = ins;
    if (iter != p->begin())
    {
        iter = std::prev(iter);
        if (iter->name() == "gpu::wait_event")
            iter = std::prev(iter);
        return (iter->name() == "gpu::set_stream") ? &(*iter) : nullptr;
    }
    return nullptr;
}

void dom_info::compute_dom(bool reversed)
{
    std::size_t num_of_instrs = p_program->size();
    if(num_of_instrs == 0)
        return;
    std::unordered_map<const instruction*, std::set<const instruction*>> instr2_doms;
    std::unordered_map<const instruction*, int> instr2_points;
    int cur_points  = reversed ? num_of_instrs - 1 : 0;
    bool seen_stream = false;
    program_visitor vis{p_program, reversed};
    std::unordered_map<const instruction*, const instruction*>& instr2_dom_tree =
        (reversed ? instr2_ipdom : instr2_idom);
    for(auto ins = vis.begin(), end = vis.end();; ins = vis.next(ins))
    {
        const instruction* p_ins = &(*ins);
        instr2_points[p_ins]     = cur_points;
        if(get_stream(p_program, ins) == nullptr)
        {
            if(reversed)
                cur_points--;
            else
                cur_points++;
            ;
            if(ins == end)
                break;
            continue;
        }
        seen_stream               = true;
        const instruction* p_tmp = nullptr;
        int cnt                  = 0;
        // find dominators.
        for(auto&& iter : vis.get_inputs(ins))
        {
            if(get_stream(p_program, iter) == nullptr)
                continue;
            const instruction * p_arg = &(*iter);
            cnt++;
            assert(instr2_doms.find(p_arg) != instr2_doms.end());
            if(p_tmp == nullptr)
                instr2_doms[p_ins] = instr2_doms[p_arg];
            else
                instr2_doms[p_ins] =
                    set_op::set_intersection(instr2_doms[p_ins], instr2_doms[p_arg]);
            p_tmp = p_arg;
        }
        // find immediate dominators.
        if(cnt == 1)
        {
            instr2_dom_tree[p_ins] = p_tmp;
        }
        else if(cnt > 0)
        {
            for(auto& iter1 : instr2_doms[p_ins])
            {
                std::unordered_map<const instruction*, const instruction*>& idom =
                    reversed ? instr2_ipdom : instr2_idom;
                auto dom_check = [& dom_tree = idom, ins1 = iter1](const instruction* ins2) {
                    if(ins1 != ins2)
                    {
                        const instruction* iter = ins2;
                        ;
                        while(dom_tree.find(iter) != dom_tree.end())
                        {
                            if(ins1 == dom_tree[iter])
                                return true;
                            iter = dom_tree[iter];
                        }
                    }
                    return false;
                };

                // check whether iter1 strictly dominates or post-dominates any other notes in
                // p_ins's dominators or post-dominators.
                if(!std::any_of(instr2_doms[p_ins].begin(), instr2_doms[p_ins].end(), dom_check))
                {
                    assert(instr2_dom_tree.find(p_ins) == instr2_dom_tree.end());
                    instr2_dom_tree[p_ins] = iter1;
                }
            }
        }

        instr2_doms[p_ins].insert(p_ins);
        if(ins == end)
            break;
        if(reversed)
            cur_points--;
        else
            cur_points++;
    }
    if(seen_stream)
    {
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
        if(!post_dom && (instr2_idom.find(p_ins) != instr2_idom.end()))
        {
            const instruction* idom = instr2_idom[p_ins];
            std::cout << "@" << instr2_points[p_ins] << " imm dominator: "
                      << "@" << instr2_points[idom] << std::endl;
        }
        if(post_dom && (instr2_ipdom.find(p_ins) != instr2_ipdom.end()))
        {
            const instruction* ipdom = instr2_ipdom[p_ins];
            std::cout << "@" << instr2_points[p_ins] << " imm post domimator: "
                      << "@" << instr2_points[ipdom] << std::endl;
        }
    }
}
#endif
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
