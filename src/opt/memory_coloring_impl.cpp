#include "memory_coloring_impl.hpp"
#include <set>

namespace migraph {
void memory_coloring_impl::run()
{
    int num_of_instrs = p_program->get_size();
    if (num_of_instrs == 0)
        return;

    DEBUG(dump("before memory coloring"));
    int cur_points = num_of_instrs * 2;
    instruction_ref iter = std::prev(p_program->end());
    instruction_ref begin = p_program->begin();
    std::vector<T_live_interval> live_intervals;
    std::vector<instruction_ref> dead_instrs;
    std::vector<int> orderings;
    int num_of_lives = 0;
    std::unordered_map<const instruction*, int> instr2Live;
    std::set<int> live_set;
    live_set.clear();
    instr2Live.clear();
    live_intervals.resize(num_of_instrs);
    do {
        const instruction* p_iter = &(*iter);
        int def_id = -1;
        if (instr2Live.find(p_iter) != instr2Live.end()) {
            def_id = instr2Live[p_iter];
            bool isLit = isLiteral(iter);
            if (isAllocate(iter) || isLit) {
                live_intervals[def_id].begin = cur_points;
                live_intervals[def_id].result = iter->result;
                live_intervals[def_id].isLiteral = isLit;
                orderings.push_back(def_id);
                live_set.erase(def_id);
            } 
        } else if (!isParam(iter) && !isOutline(iter) && !isCheckContext(iter)) {
            // dead instruction.
            dead_instrs.push_back(iter);
        }
        
        if (!iter->arguments.empty()) {
            for (auto&& arg : iter->arguments) {
                if (isParam(arg) || isOutline(arg)) {
                    continue;
                } 
                const instruction* p_arg = &(*arg);
                if (isAllocate(arg)) {
                    assert(def_id != -1);
                    live_intervals[def_id].addUse(cur_points);
                    instr2Live[p_arg] = def_id;
                } else if (instr2Live.find(p_arg) == instr2Live.end()) {
                    int id = num_of_lives++;
                    live_intervals[id].id = id;
                    live_intervals[id].end = cur_points;
                    live_intervals[id].addUse(cur_points);
                    instr2Live[p_arg] = id;
                    live_set.insert(id);
                } 
            }
        }
        cur_points -= 2;
        iter = std::prev(iter);
    } while (iter != begin);
}

#ifdef DEBUG_OPT
void memory_coloring_impl::dump(std::string str)
{
    std::cout << str << std::endl;
    std::cout << *p_program << std::endl;
}
#endif    
    
} // namespace migraph
