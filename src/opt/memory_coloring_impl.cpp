#include "memory_coloring_impl.hpp"

namespace migraph {
void memory_coloring_impl::run()
{
    int num_of_instrs = p_program->get_size();
    if (num_of_instrs == 0)
        return;

    DEBUG(dump("---Before memory coloring---"));
    int cur_points = num_of_instrs * 2;
    instruction_ref iter = std::prev(p_program->end());
    instruction_ref begin = p_program->begin();
    std::vector<T_live_interval*> live_intervals;
    std::vector<instruction_ref> dead_instrs;
    std::list<T_live_interval*> active_queue;
    int num_of_lives = 0;
    std::unordered_map<const instruction*, T_live_interval*> instr2Live;
    std::set<int> live_set;
    T_live_interval* next_def = nullptr;
    live_intervals.reserve(num_of_instrs);
    do {
        const instruction* p_iter = &(*iter);
        T_live_interval* def_interval = nullptr;
        bool isDead = false;
        if (instr2Live.find(p_iter) != instr2Live.end()) {
            def_interval = instr2Live[p_iter];
            bool isLit = isLiteral(iter);
            if (isAllocate(iter) || isLit) {
                T_live_range& range = def_interval->segments.front();
                range.begin = cur_points;
                def_interval->result = iter->result;
                def_interval->isLiteral = isLit;
                def_interval->next_enqueue_def = cur_points;
                active_queue.push_front(def_interval);
                next_def = def_interval;
                live_set.erase(def_interval->id);
            } 
        } else if (!isParam(iter) && !isOutline(iter) && !isCheckContext(iter)) {
            isDead = true;
        }

        if (!iter->arguments.empty()) {
            for (auto&& arg : iter->arguments) {
                if (isParam(arg) || isOutline(arg)) {
                    if (isOutputParam(arg))
                        isDead = false;
                    continue;
                }
                const instruction* p_arg = &(*arg);
                if (isAllocate(arg)) {
                    // input is from hip::allocate, def is considered as use
                    // and coalesce the live intervals.
                    def_interval->addUse(cur_points);
                    instr2Live[p_arg] = def_interval;
                } else if (instr2Live.find(p_arg) == instr2Live.end()) {
                    // First time see a use, create a live interval.
                    int id = num_of_lives++;
                    T_live_interval* interval = new live_interval();
                    interval->id = id;
                    interval->segments.push_back(T_live_range{-1, cur_points, -1});
                    interval->addUse(cur_points);
                    instr2Live[p_arg] = interval;
                    live_set.insert(id);
                    // Keep track of live intervals that are inactive when
                    // next_def is enqueued.
                    if (next_def != nullptr)
                        next_def->inactive_afters.push_back(interval);
                    live_intervals[id] = interval;
                } else {
                    T_live_interval* interval = instr2Live[p_arg];
                    interval->addUse(cur_points);
                    DEBUG(assert(live_set.find(interval->id) != live_set.end()));
                }
            }
        }
        if (isDead)
            dead_instrs.push_back(iter);
        cur_points -= 2;
        iter = std::prev(iter);
    } while (iter != begin);

    DEBUG(dump(live_intervals, num_of_lives));
    for (int i = 0; i < num_of_lives; ++i)
        free(live_intervals[i]);
}

#ifdef DEBUG_OPT
void memory_coloring_impl::dump(std::string str)
{
    std::cout << str << std::endl;
    std::cout << *p_program << std::endl;
}

void memory_coloring_impl::dump(std::vector<T_live_interval*>& live_intervals, int num_of_lives)
{
    if (num_of_lives > 0) {
        std::cout << "---live intervals ---" << std::endl;
        for (int i = 0; i < num_of_lives; ++i) {
            T_live_interval* interval = live_intervals[i];
            interval->dump();
        }
    }
}

#define GET_INS_ENUM(x) (((x) >> 1) - 1)
void live_interval::dump()
{

    std::cout << "id:" << id;
    for (auto iter = segments.begin(), end = segments.end(); iter != end; ++iter) {
        T_live_range& range = *iter;
        std::cout << " [" << GET_INS_ENUM(range.begin)  << ", " << GET_INS_ENUM(range.end) << "]";
    }

    std::cout << " uses:";
    for (auto iter = use_points.begin(), end = use_points.end(); iter != end; ++iter) {
        int& use = *iter;
        std::cout << " " << GET_INS_ENUM(use) << ",";
    }
    if (!inactive_afters.empty()) {
        std::cout << " inactivate:";
        for (auto iter = inactive_afters.begin(), end = inactive_afters.end(); iter != end; ++iter) {
            T_live_interval*& interval = *iter;
            std::cout << " " << interval->id << ",";
        }
    }
    if (isLiteral)
        std::cout << " literal";
    std::cout << std::endl;
}
#endif    
    
} // namespace migraph
