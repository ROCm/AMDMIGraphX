#include <migraphx/dom_info.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// A unified interface to visit programs top-down or bottom-up.
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

// Query whether ins1 strictly dominates ins2.  ins1 strictly dominates
// ins2 if ins1 dominates ins2 and ins1 is not ins2.
//
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

// Query whether ins1 strictly post-dominates ins2.  ins1 strictly post-dominates
// ins2 if ins1 post-dominates ins2 and ins1 is not ins2.
//
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

//  Among p_ins's dominators, find ones that strictly dominates or post-dominators others.
//
void dom_info::find_dom_tree(
    std::unordered_map<const instruction*, std::set<const instruction*>>& instr2_doms,
    const instruction* p_ins,
    std::unordered_map<const instruction*, const instruction*>& instr2_dom_tree,
    std::unordered_map<const instruction*, const instruction*>& idom)
{
    for(auto& iter1 : instr2_doms[p_ins])
    {
        auto dom_check = [& dom_tree = idom, ins1 = iter1 ](const instruction* ins2)
        {
            if(ins1 == ins2)
                return false;
            const instruction* iter = ins2;
            ;
            while(dom_tree.find(iter) != dom_tree.end())
            {
                if(ins1 == dom_tree[iter])
                    return true;
                iter = dom_tree[iter];
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

//  Compute dominator or post-dominator.  Instructions that do not use
//  streams are left out.
//
void dom_info::compute_dom(bool reversed)
{
    std::size_t num_of_instrs = p_program->size();
    if(num_of_instrs == 0)
        return;
    std::unordered_map<const instruction*, std::set<const instruction*>> instr2_doms;
    std::unordered_map<const instruction*, int> instr2_points;
    int cur_points   = reversed ? num_of_instrs - 1 : 0;
    bool seen_stream = false;
    program_visitor vis{p_program, reversed};
    std::unordered_map<const instruction*, const instruction*>& instr2_dom_tree =
        (reversed ? instr2_ipdom : instr2_idom);
    for(auto ins = vis.begin(), end = vis.end();; ins = vis.next(ins))
    {
        const instruction* p_ins = &(*ins);
        instr2_points[p_ins]     = cur_points;
        if(ins->get_stream() < 0)
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
        seen_stream              = true;
        const instruction* p_tmp = nullptr;
        int cnt                  = 0;
        // find dominators.
        for(auto&& iter : vis.get_inputs(ins))
        {
            if(iter->get_stream() < 0)
                continue;
            const instruction* p_arg = &(*iter);
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
            std::unordered_map<const instruction*, const instruction*>& idom =
                reversed ? instr2_ipdom : instr2_idom;
            find_dom_tree(instr2_doms, p_ins, instr2_dom_tree, idom);
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

// Identify split points.  A split point has more than one
// outputs that are executed in different streams.

bool dom_info::is_split_point(instruction_ref ins)
{
    if(ins->has_mask(record_event))
    {
        std::set<int> stream_set;
        for(auto&& arg : ins->outputs())
        {
            int arg_stream = arg->get_stream();
            if(arg_stream >= 0)
                stream_set.insert(arg_stream);
        }
        if(stream_set.size() > 1)
            return true;
    }
    return false;
}

// Identify merge points.  A merge point has more than one
// inputs that are executed in different streams.
bool dom_info::is_merge_point(instruction_ref ins)
{
    if(ins->has_mask(wait_event))
    {
        std::set<int> stream_set;
        for(auto&& arg : ins->inputs())
        {
            int arg_stream = arg->get_stream();
            if(arg_stream >= 0)
                stream_set.insert(arg_stream);
        }
        if(stream_set.size() > 1)
            return true;
    }
    return false;
}

//  Propagate split points through the graph and identify concurrent instructions.
//  Concurrent instructions have the same split points and different streams.
//
void dom_info::propagate_splits(
    int num_of_streams,
    std::unordered_map<const instruction*, std::vector<std::vector<const instruction*>>>&
        concur_instrs,
    std::unordered_map<const instruction*, int>& instr2_points)
{
    std::unordered_map<instruction_ref, bool> is_split;
    std::unordered_map<instruction_ref, bool> is_merge;
    std::unordered_map<instruction_ref, std::set<const instruction*>> split_from;
    int cur_points = 0;
    instr2_points.clear();

    for(auto ins : iterator_for(*p_program))
    {
        const instruction* p_iter = &(*ins);
        instr2_points[p_iter]     = cur_points++;
        int stream                = ins->get_stream();
        if(stream < 0)
            continue;

        is_split[ins] = is_split_point(ins);
        is_merge[ins] = is_merge_point(ins);

        for(auto&& arg : ins->inputs())
        {
            // Input is a split point.
            if(is_split.find(arg) != is_split.end())
                split_from[ins].insert(&(*arg));
            // Union inputs' split points.
            if((split_from.find(arg) != split_from.end()) && !split_from[arg].empty())
            {
                if(split_from.find(ins) == split_from.end())
                    split_from[ins] = split_from[arg];
                else
                    split_from[ins] = set_op::set_union(split_from[ins], split_from[arg]);
            }
        }

        if(is_merge[ins])
        {
            assert(split_from.find(ins) != split_from.end());
            std::set<const instruction*> del_set;
            // post-dominator kills split point.
            for(auto& split : split_from[ins])
            {
                if(strictly_post_dominates(p_iter, split))
                    del_set.insert(split);
            }
            split_from[ins] = set_op::set_difference(split_from[ins], del_set);
        }

        if(split_from.find(ins) != split_from.end())
        {
            // Collect concur instructions for each split point.
            for(auto& split : split_from[ins])
            {
                if(concur_instrs.find(split) == concur_instrs.end())
                {
                    std::vector<std::vector<const instruction*>> instr_stack;
                    instr_stack.resize(num_of_streams);
                    concur_instrs[split] = instr_stack;
                }
                concur_instrs[split][stream].push_back(p_iter);
            }
        }
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
