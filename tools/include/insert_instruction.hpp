#ifndef MIGRAPH_GUARD_INSERT_INSTRUCTION_HPP
#define MIGRAPH_GUARD_INSERT_INSTRUCTION_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace migraphx {

#ifdef DOXYGEN

/// An interface for target-dependent instruction insertion.
/// executing in different streams.
struct insert_instruction
{
    void
    insert_event(program* p,
                 int mask,
                 instruction_ref ins,
                 std::vector<instruction_ref> args);
};

#else

<%
interface('insert_instruction',
          virtual('insert_event', returns='void', p = 'program*', mask = 'int', ins = 'instruction_ref', input = 'std::vector<instruction_ref>')
)
%>

#endif
} // namespace migraphx

#endif
