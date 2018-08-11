#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_IMPL_HPP
#include "common_header.hpp"

namespace migraph {
struct memory_coloring_impl {
    explicit memory_coloring_impl(program *p) : p_program(p) {}
    void run();
    private:
    bool isParam(const instruction_ref ins)    { return ins->op.name() == "@param"; }
    bool isAllocate(const instruction_ref ins) { return ins->op.name() == "hip::allocate"; }
    bool isOutline(const instruction_ref ins)  { return ins->op.name() == "@outline"; }
    bool isLiteral(const instruction_ref ins)  { return ins->op.name() == "@literal"; }
    bool isCheckContext(const instruction_ref ins) { return ins->op.name() == "check_context"; }

#ifdef DEBUG_OPT    
    void dump(std::string);
#endif    

    program* p_program;
};

typedef struct live_interval {
    explicit live_interval() { init(); }
    bool isValid()  const    { return (begin != -1) && (end != -1); }
    void addUse(int use)     { use_points.push_back(use); }
    void init() { begin = -1; end = -1; id = -1; isLiteral = false; }
    int begin;
    int end;
    int id;
    std::vector<int> use_points;
    shape result;
    bool isLiteral;
} T_live_interval;

} // namespace migraph
#endif
