#include "memory_coloring_impl.hpp"

namespace migraph {
void memory_coloring_impl::run()
{
    build();
    if (num_of_lives != 0) {
        DEBUG(dump("---Before memory coloring---"));
        DEBUG(dump());
        // Coloring
        while (!alloc_queue.empty()) {
            T_live_interval* interval = alloc_queue.top();
            allocate(interval);
            alloc_queue.pop();
        }
        for (int i = 0; i < num_of_lives; ++i)
            free(live_intervals[i]);
    }
}

bool memory_coloring_impl::allocate(T_live_interval* interval)
{
    shape s = interval->result;
    int size = s.bytes();
    std::size_t element_size = size / s.elements();
    T_live_range& segment = interval->segment;
    int vn = segment.vn;
    std::priority_queue<T_live_range*, std::vector<T_live_range*>, ordering> conflict_queue;

    if (conflict_table.find(vn) != conflict_table.end()) {
        std::set<int>& vn_set = conflict_table[vn];
        for (auto iter = vn_set.begin(), end = vn_set.end(); iter != end; ++iter) {
            T_live_range* range = live_ranges[*iter];
            if (range->offset != -1)
                conflict_queue.push(range);
        }
    }

    int offset = 0;
    while (!conflict_queue.empty()) {
        T_live_range* range = conflict_queue.top();
        int cur_offset = range->offset;
        if ((cur_offset > offset) && (cur_offset - offset) >= size) {
            break;
        }
        offset = cur_offset + range->size;
        if ((offset % element_size) != 0)
            offset += (element_size - (offset % element_size));
        conflict_queue.pop();
    }
    segment.offset = offset;
    return true;
}

void memory_coloring_impl::build()
{
    int num_of_instrs = p_program->get_size();
    if (num_of_instrs == 0)
        return;
    int cur_points = num_of_instrs * 2;
    instruction_ref iter = std::prev(p_program->end());
    instruction_ref begin = p_program->begin();
    std::vector<instruction_ref> dead_instrs;
    std::unordered_map<const instruction*, T_live_interval*> instr2Live;
    std::set<int> live_set;
    T_live_interval* next_def = nullptr;
    // Build live intervals.
    do {
        const instruction* p_iter = &(*iter);
        T_live_interval* def_interval = nullptr;
        bool isDead = false;
        if (instr2Live.find(p_iter) != instr2Live.end()) {
            def_interval = instr2Live[p_iter];
            bool isLit = isLiteral(iter);
            if (isAllocate(iter) || isLit) {
                T_live_range& range = def_interval->segment;
                def_interval->result = iter->result;
                def_interval->isLiteral = isLit;
                alloc_queue.push(def_interval);
                range.begin = cur_points;
                range.size = (iter->result).bytes();
                next_def = def_interval;
                live_set.erase(range.vn);
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
                    interval->segment.end = cur_points;
                    interval->segment.vn = ++max_value_number;
                    interval->addUse(cur_points);
                    instr2Live[p_arg] = interval;
                    addConflicts(live_set, max_value_number);
                    live_set.insert(max_value_number);
                    live_intervals[id] = interval;
                    live_ranges[max_value_number] = &(interval->segment);
                    // Keep track of live intervals that are inactive when
                    // next_def is enqueued.
                    if (next_def != nullptr)
                        next_def->inactive_afters.push_back(interval);
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
}
 
#ifdef DEBUG_OPT
void memory_coloring_impl::dump(std::string str)
{
    std::cout << str << std::endl;
    std::cout << *p_program << std::endl;
}

void memory_coloring_impl::dump()
{
    if (num_of_lives > 0) {
        std::cout << "---live intervals ---" << std::endl;
        for (int i = 0; i < num_of_lives; ++i) {
            T_live_interval* interval = live_intervals[i];
            interval->dump();
        }
        std::cout << "conflict table:" << std::endl;
        for (int i = 0; i <= max_value_number; ++i) {
            std::cout << " segment:" << i;
            std::cout << " =>";
            std::set<int>& table = conflict_table[i];
            for (auto iter = table.begin(), end = table.end(); iter != end; ++iter) {
                std::cout << (*iter) << ",";
            }
        }
        std::cout << std::endl;
    }
}

#define GET_INS_ENUM(x) (((x) >> 1) - 1)
void live_interval::dump()
{
    std::cout << "id:" << id;
    std::cout << " segment:" << segment.vn;
    std::cout << " [" << GET_INS_ENUM(segment.begin)  << ", " << GET_INS_ENUM(segment.end) << "]";
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
    std::cout << " " << result;
    std::cout << std::endl;
}
#endif    
    
} // namespace migraph
