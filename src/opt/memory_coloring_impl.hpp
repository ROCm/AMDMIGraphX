#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#include "common_header.hpp"

namespace migraph {

typedef struct live_range {
    int begin;  // begin point in the instruction stream.
    int end;    // end point in the instruction stream.
    int offset; // offset to base pointer of allocated memory trunk.
    int vn;     // value number that identifies this live_range.
    int size;   // size of required memory in bytes
#ifdef DEBUG_OPT
    void dump();
#endif    
} T_live_range;
    
typedef struct live_interval {
    explicit live_interval() { init(); }
    
    void init() {
        id = -1; isLiteral = false;
        segment = { -1, -1, -1, -1, 0};
    }
    void addUse(int use)     { use_points.push_front(use); }
    int get_begin() const { return segment.begin; }
    int get_end()   const { return segment.end; }

#ifdef DEBUG_OPT
    void dump();
#endif    

    T_live_range segment;
    int id;
    std::list<int> use_points;
    // Live intervals that are inactive when this live interval is enqueued.
    // can be used for live interval collapsing.
    std::list<struct live_interval*> inactive_afters;
    shape result;
    bool isLiteral;

} T_live_interval;

struct memory_coloring_impl {
    explicit memory_coloring_impl(program *p) : p_program(p)
    {
        live_intervals.clear();
        live_ranges.clear();
        conflict_table.clear();
        num_of_lives = 0;
        max_value_number = -1;
    }
    bool allocate(T_live_interval*);
    void addConflicts(std::set<int>& live_set, int val)
    {
        for (auto iter = live_set.begin(), end = live_set.end(); iter != end; ++ iter) {
            conflict_table[*iter].insert(val);
            conflict_table[val].insert(*iter);
        }
    }
    void build();
    void run();
    private:
    bool isParam(const instruction_ref ins)    { return ins->op.name() == "@param"; }
    bool isOutputParam(const instruction_ref ins)
    {
        return isParam(ins) && any_cast<builtin::param>(ins->op).parameter == "output";
    }
    bool isAllocate(const instruction_ref ins) { return ins->op.name() == "hip::allocate"; }
    bool isOutline(const instruction_ref ins)  { return ins->op.name() == "@outline"; }
    bool isLiteral(const instruction_ref ins)  { return ins->op.name() == "@literal"; }
    bool isCheckContext(const instruction_ref ins) { return ins->op.name() == "check_context"; }
    
#ifdef DEBUG_OPT
    void dump(std::string);
    void dump();
#endif    
    struct ordering { 
        bool operator() (const T_live_interval* I1, const T_live_interval* I2) const
        {
            int len1 = I1->get_end() - I1->get_begin();
            int len2 = I2->get_end() - I2->get_begin();
            if (len1 != len2) {
                return (len1 < len2) ? true : false;
            } else if (I1->result.bytes() != I2->result.bytes()) {
                return (I1->result.bytes() < I2->result.bytes()) ? true : false;
            } else {
                return I1->id > I2->id;
            }
        }
        bool operator() (const T_live_range* I1, const T_live_range* I2) const
        {
            return (I1->offset > I2->offset);
        }
    };
    program* p_program;
    // Map live interval Id to live interval.
    std::unordered_map<int, T_live_interval*> live_intervals;
    // Map live range value number to live range.
    std::unordered_map<int, T_live_range*> live_ranges;
    // Map live range value number to a set of conflicting live ranges' value numbers.
    std::unordered_map<int, std::set<int>> conflict_table;
    // Priority queue for coloring.
    std::priority_queue<T_live_interval*, std::vector<T_live_interval*>, ordering> alloc_queue;
    
    int num_of_lives;
    int max_value_number;
};    

} // namespace migraph
#endif
