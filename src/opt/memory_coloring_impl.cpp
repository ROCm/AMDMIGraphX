#include "memory_coloring_impl.hpp"

namespace migraph {
void memory_coloring_impl::run()
{
    build();
    if (num_of_lives != 0) {
        DEBUG(dump("---Before memory coloring---"));
        DEBUG(dump(p_program));
        // Coloring
        while (!alloc_queue.empty()) {
            T_live_interval* interval = alloc_queue.top();
            allocate(interval);
            alloc_queue.pop();
        }
        rewrite();
        DEBUG(verify());
        for (int i = 0; i < num_of_lives; ++i) {
            free(live_intervals[i]);
        }
    }
}

bool memory_coloring_impl::allocate(T_live_interval* interval)
{
    shape s = interval->result;
    std::size_t size = s.bytes();
    std::size_t element_size = size / s.elements();
    T_live_range& segment = interval->segment;
    int vn = segment.vn;
    std::priority_queue<T_live_range*, std::vector<T_live_range*>, ordering> conflict_queue;
    std::unordered_map<long long, T_live_range*> offset2Live;
    offset2Live.clear();    

    if (conflict_table.find(vn) != conflict_table.end()) {
        std::set<int>& vn_set = conflict_table[vn];
        for (auto iter = vn_set.begin(), end = vn_set.end(); iter != end; ++iter) {
            T_live_range* range = live_ranges[*iter];
            long long offset = range->offset;
            if (offset != InvalidOffset) {
                conflict_queue.push(range);
                if (offset2Live.find(offset) == offset2Live.end()) {
                    offset2Live[offset] = range;
                } else {
                    T_live_range* prev = offset2Live[offset];
                    assert(prev->offset == offset);
                    if (prev->size < range->size)
                        offset2Live[offset] = range;
                }
            }
        }
    }

    long long offset = 0;
    while (!conflict_queue.empty()) {
        T_live_range* range = conflict_queue.top();
        long long cur_offset = range->offset;
        if (offset2Live[cur_offset] == range) {
            if ((cur_offset > offset) && (cur_offset - offset) >= size) {
                break;
            }
            offset = cur_offset + range->size;
            if ((offset % element_size) != 0)
                offset += (element_size - (offset % element_size));
        }
        conflict_queue.pop();
    }
    segment.offset = offset;
    DEBUG(segment.dump());
    required_bytes = std::max(required_bytes, offset + segment.size);
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
    std::set<int> live_set;
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
                live_set.erase(range.vn);
            } 
        } else if (!isParam(iter) && !isOutline(iter) && !isCheckContext(iter)) {
            isDead = true;
        }
        int tieNdx = getInputTieNdx(iter);
        if (!iter->arguments.empty()) {
            int cnt = -1;
            for (auto&& arg : iter->arguments) {
                cnt++;
                if (isParam(arg) || isOutline(arg)) {
                    if (isOutputParam(arg))
                        isDead = false;
                    continue;
                }
                const instruction* p_arg = &(*arg);
                if (cnt == tieNdx) {
                    // input memory is used as this instruction's output.
                    // def is considered as use. Coalesce the live intervals.
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

void memory_coloring_impl::rewrite()
{
    instruction_ref end = p_program->end();
    instruction_ref scratch_param = end;
    for (auto ins : iterator_for(*p_program)) {
        const instruction* p_iter = &(*ins);
        if (isScratchParam(ins)) {
            scratch_param = ins;
            int allocated_bytes = ins->result.bytes();
            if (allocated_bytes < required_bytes) {
                std::cout << "required bytes: " << required_bytes << "allocated bytes: " << allocated_bytes << std::endl;
                throw std::runtime_error("insufficent memory for MIGraph");
            }
#ifdef DEBUG_OPT
            float frac = 1.0 * required_bytes/allocated_bytes*100;
            std::cout << "memory usage percentage: " << to_string(frac) << "%" << std::endl;
#endif            
        }
        if (instr2Live.find(p_iter) != instr2Live.end()) {
            T_live_interval* interval = instr2Live[p_iter];
            if (interval->get_offset() == InvalidOffset) {
                DEBUG(assert(interval->get_begin() == InvalidOffset));
                continue;
            }
            std::size_t offset = interval->get_offset();
            if (isAllocate(ins)) {
                if (scratch_param == end)
                    throw std::runtime_error("missing scratch parameter");
                p_program->replace_instruction(ins, get_mem_ptr{offset}, scratch_param, ins->arguments.at(0));
            } else if (isLiteral(ins)) {
                if (scratch_param == end)
                    throw std::runtime_error("missing scratch parameter");
                auto pre = p_program->add_literal(ins->lit);
                auto index = p_program->add_literal(offset);
                p_program->replace_instruction(ins, write_literal{}, scratch_param, index, pre);
            }
        }
    }
    DEBUG(dump("---After rewrite---"));
    DEBUG(dump(p_program));
}
 
#ifdef DEBUG_OPT
void memory_coloring_impl::dump(std::string str)
{
    std::cout << str << std::endl;
    
}

void memory_coloring_impl::dump(program* p_program)
{
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
        std::cout << "---conflict table---" << std::endl;
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

void memory_coloring_impl::verify()
{
    if (num_of_lives > 0) {
        for (int i = 0; i < num_of_lives; ++i) {
            T_live_interval* interval = live_intervals[i];
            T_live_range& segment = interval->segment;
            if (segment.offset == InvalidOffset)
                continue;
            int vn = segment.vn;
            if (conflict_table.find(vn) != conflict_table.end()) {
                std::set<int>& vn_set = conflict_table[vn];
                for (auto iter = vn_set.begin(), end = vn_set.end(); iter != end; ++iter) {
                    T_live_range* range = live_ranges[*iter];
                    if (range->offset == InvalidOffset)
                        continue;
                    if (!isDisjoin(*range, segment))
                        assert(false);
                }
            }
        }
    }
}

#define GET_INS_ENUM(x) (((x) >> 1) - 1)
    
void live_range::dump()
{
    std::cout << " segment:" << vn;
    std::cout << " [" << GET_INS_ENUM(begin)  << ", " << GET_INS_ENUM(end) << "]";
    if (offset != InvalidOffset) {
        std::cout << " mem:";
        std::cout << " [" << offset << "," << offset + size - 1 << "]";
    }
    std::cout << std::endl;
}
    
void live_interval::dump()
{
    std::cout << "id:" << id;
    segment.dump();
    std::cout << " uses:";
    for (auto iter = use_points.begin(), end = use_points.end(); iter != end; ++iter) {
        int& use = *iter;
        std::cout << " " << GET_INS_ENUM(use) << ",";
    }

    if (isLiteral)
        std::cout << " literal";
    std::cout << " " << result;
    std::cout << std::endl;
}
    
#endif    
    
} // namespace migraph
