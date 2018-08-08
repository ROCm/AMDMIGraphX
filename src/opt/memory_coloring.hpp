#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP
#include "common_header.hpp"

namespace migraph {
struct memory_coloring {
    explicit memory_coloring(program *p) : p_program(p) {}
    void run();
    private:
    program* p_program;
};

} // namespace migraph
#endif
