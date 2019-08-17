#ifndef MIGRAPHX_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_config.hpp>
#include <migraphx/config.hpp>

#include <set>
#include <list>
#include <vector>
#include <queue>

#ifdef MIGRAPHX_DEBUG_OPT
#define MIGRAPHX_DEBUG(s) s
#else
#define MIGRAPHX_DEBUG(s)
#endif // MIGRAPHX_DEBUG_OPT

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static const std::size_t invalid_offset = std::numeric_limits<std::size_t>::max();

struct live_range
{
    std::size_t begin;  // begin point in the instruction stream.
    std::size_t end;    // end point in the instruction stream.
    std::size_t offset; // offset to base pointer of allocated memory trunk.
    std::size_t vn;     // value number that identifies this live_range.
    std::size_t size;   // size of required memory in bytes
#ifdef MIGRAPHX_DEBUG_OPT
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

    void add_use(std::size_t use) { use_points.push_front(use); }
    std::size_t get_begin() const { return segment.begin; }
    std::size_t get_end() const { return segment.end; }
    long long get_offset() const { return segment.offset; }

#ifdef MIGRAPHX_DEBUG_OPT
    void dump();
#endif

    live_range segment;
    std::size_t id;
    std::list<std::size_t> use_points;
    std::size_t def_point;
    shape result;
    bool is_literal;
    bool is_live_on_entry;
};

using interval_ptr = live_interval*;

struct memory_coloring_impl
{
    memory_coloring_impl(program* p, std::string alloc_op, bool p_verify)
        : p_program(p), allocation_op(std::move(alloc_op)), enable_verify(p_verify)
    {
        instr2_live.clear();
        live_ranges.clear();
        conflict_table.clear();
        num_of_lives       = 0;
        max_value_number   = -1;
        required_bytes     = 0;
        earliest_end_point = -1;
        latest_end_point   = -1;
        unify_literals     = false;
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
    void rewrite();

    private:
    static bool is_param(const instruction_ref ins) { return ins->name() == "@param"; }
    static bool is_output_param(const instruction_ref ins)
    {
        return is_param(ins) && any_cast<builtin::param>(ins->get_operator()).parameter == "output";
    }
    bool is_allocate(const instruction_ref ins) const { return ins->name() == allocation_op; }
    static bool is_outline(const instruction_ref ins) { return ins->name() == "@outline"; }
    static bool is_literal(const instruction_ref ins) { return ins->name() == "@literal"; }
    static bool is_check_context(const instruction_ref ins)
    {
        return ins->name() == "check_context";
    }

    static bool is_disjoin(const live_range& range1, const live_range& range2)
    {
        if((range1.size == 0) || (range2.size == 0))
            return false;
        auto end1 = range1.offset + range1.size - 1;
        auto end2 = range2.offset + range2.size - 1;
        return ((end1 < range2.offset) || (end2 < range1.offset));
    }
    void verify();
#ifdef MIGRAPHX_DEBUG_OPT
    void dump(const std::string&);
    void dump_program();
    void dump_intervals();
#endif
    struct ordering
    {
        bool operator()(const interval_ptr i1, const interval_ptr i2) const
        {
            auto len1 = i1->get_end() - i1->get_begin();
            auto len2 = i2->get_end() - i2->get_begin();
            if(len1 != len2)
            {
                return (len1 < len2);
            }
            else if(i1->result.bytes() != i2->result.bytes())
            {
                return (i1->result.bytes() < i2->result.bytes());
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

    int num_of_lives;
    int max_value_number;
    std::size_t required_bytes;
    // The earliest program point where an live interval ends.
    int earliest_end_point;
    // The latest program point where an live interval ends.
    int latest_end_point;
    // Whether to unify literals into coloring.
    bool unify_literals;
    std::string allocation_op{};
    bool enable_verify;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
