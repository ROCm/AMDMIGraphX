#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#include "common_header.hpp"

namespace migraph {

static const int invalid_offset = -1;

struct live_range
{
    int begin;        // begin point in the instruction stream.
    int end;          // end point in the instruction stream.
    long long offset; // offset to base pointer of allocated memory trunk.
    int vn;           // value number that identifies this live_range.
    long long size;   // size of required memory in bytes
#ifdef MIGRAPH_DEBUG_OPT
    void dump();
#endif
};

struct live_interval
{
    live_interval() : segment({invalid_offset, invalid_offset, invalid_offset, invalid_offset, 0})
    {
        id               = invalid_offset;
        def_point        = invalid_offset;
        is_literal       = false;
        is_live_on_entry = false;
    }

    void add_use(int use) { use_points.push_front(use); }
    int get_begin() const { return segment.begin; }
    int get_end() const { return segment.end; }
    long long get_offset() const { return segment.offset; }

#ifdef MIGRAPH_DEBUG_OPT
    void dump();
#endif

    live_range segment;
    int id;
    std::list<int> use_points;
    int def_point;
    shape result;
    bool is_literal;
    bool is_live_on_entry;
};

using interval_ptr = live_interval*;

struct memory_coloring_impl
{
    memory_coloring_impl(program* p, std::string alloc_op) : p_program(p), allocation_op(std::move(alloc_op))
    {
        instr2_live.clear();
        live_ranges.clear();
        conflict_table.clear();
        num_of_lives     = 0;
        max_value_number = -1;
        required_bytes   = 0;
        operand_alias.clear();
        earliest_end_point = -1;
    }
    bool allocate(interval_ptr);
    void add_conflicts(std::set<int>& live_set, int val)
    {
        for(auto& iter : live_set)
        {
            conflict_table[iter].insert(val);
            conflict_table[val].insert(iter);
        }
    }
    void build();
    void run();
    void register_operand_alias();
    void rewrite();

    private:
    static bool is_param(const instruction_ref ins) { return ins->op.name() == "@param"; }
    static bool is_output_param(const instruction_ref ins)
    {
        return is_param(ins) && any_cast<builtin::param>(ins->op).parameter == "output";
    }
    bool is_allocate(const instruction_ref ins) { return ins->op.name() == allocation_op; }
    static bool is_outline(const instruction_ref ins) { return ins->op.name() == "@outline"; }
    static bool is_literal(const instruction_ref ins) { return ins->op.name() == "@literal"; }
    static bool is_check_context(const instruction_ref ins)
    {
        return ins->op.name() == "check_context";
    }

    // get operand alias info.  This is a temporary workaround.
    int get_input_tie_ndx(const instruction_ref ins)
    {
        std::string name = ins->op.name();
        if(operand_alias.find(name) != operand_alias.end())
            return operand_alias[name];
        if(is_allocate(ins))
        {
            // This happens to custom allocators.
            operand_alias[name] = -1;
            return -1;
        }
        int cnt           = -1;
        int last_allocate = -1;
        for(auto&& arg : ins->arguments)
        {
            cnt++;
            if(is_allocate(arg) || is_output_param(arg))
                last_allocate = cnt;
        }
        assert(last_allocate != -1);
        operand_alias[name] = last_allocate;
        return last_allocate;
    }
#ifdef MIGRAPH_DEBUG_OPT
    static bool is_disjoin(live_range& range1, live_range& range2)
    {
        if((range1.size == 0) || (range2.size == 0))
            return false;
        long long end1 = range1.offset + range1.size - 1;
        long long end2 = range2.offset + range2.size - 1;
        return ((end1 < range2.offset) || (end2 < range1.offset));
    }
    void dump(const std::string&);
    void dump_program();
    void dump_intervals();
    void verify();
#endif
    struct ordering
    {
        bool operator()(const interval_ptr i1, const interval_ptr i2) const
        {
            int len1 = i1->get_end() - i1->get_begin();
            int len2 = i2->get_end() - i2->get_begin();
            if(len1 != len2)
            {
                return (len1 < len2) ? true : false;
            }
            else if(i1->result.bytes() != i2->result.bytes())
            {
                return (i1->result.bytes() < i2->result.bytes()) ? true : false;
            }
            else
            {
                return i1->id > i2->id;
            }
        }
        bool operator()(const live_range* i1, const live_range* i2) const
        {
            return (i1->offset > i2->offset);
        }
    };
    program* p_program;
    std::unordered_map<const instruction*, interval_ptr> instr2_live;
    // universe of live intervals.
    std::vector<live_interval> live_intervals;
    // Map live range value number to live range.
    std::unordered_map<int, live_range*> live_ranges;
    // Map live range value number to a set of conflicting live ranges' value numbers.
    std::unordered_map<int, std::set<int>> conflict_table;
    // Priority queue for coloring.
    std::priority_queue<interval_ptr, std::vector<interval_ptr>, ordering> alloc_queue;
    std::unordered_map<std::string, int> operand_alias;

    int num_of_lives;
    int max_value_number;
    long long required_bytes;
    // The earliest program point where an live interval ends.
    int earliest_end_point;
    std::string allocation_op{};
};
} // namespace migraph
#endif
