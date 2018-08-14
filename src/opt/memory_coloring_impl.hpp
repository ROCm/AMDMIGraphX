#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#include "common_header.hpp"

namespace migraph {

typedef struct live_range {
    explicit live_range(int b, int e, int o ) : begin(b), end(e), offset(o) {};
    int begin;
    int end;
    int offset;
} T_live_range;
    
typedef struct live_interval {
    explicit live_interval() { init(); }
    void addUse(int use)     { use_points.push_front(use); }
    void init() {
        id = -1; isLiteral = false;
    }
    std::list <T_live_range> segments;
    int id;
    std::list<int> use_points;
    // Live intervals that are inactive when this live interval is enqueued.
    std::list<struct live_interval*> inactive_afters;
    // Next enqueue point for this live interval.  It is not always
    // equal to the begin if this live interval is rematerialized.
    int next_enqueue_def;
    shape result;
    bool isLiteral;

#ifdef DEBUG_OPT
    void dump();
#endif    
} T_live_interval;

typedef struct occupant_range {
    explicit occupant_range(int b, int e, T_live_interval* in)
        : begin(b), end(e), interval(in) {};
    int begin;
    int end;
    T_live_interval* interval;
} T_occupant_range;

struct memory_coloring_impl {
    explicit memory_coloring_impl(program *p) : p_program(p) {}
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
    void dump(std::vector<T_live_interval*>&, int);
#endif    

    program* p_program;
};    

} // namespace migraph
#endif
