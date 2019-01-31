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
/// for multi-stream execution.
struct insert_instruction
{
    void insert_create_events(program* p, instruction_ref ins, int num_of_events);
    void insert_record_event(program* p, instruction_ref ins, int event);
    void insert_wait_event(program* p, instruction_ref ins, int event);

    void insert_stream(program* p, instruction_ref ins, int stream);
};

#else

<%
interface('insert_instruction',
          virtual('insert_create_events', returns='void', p = 'program*', ins ='instruction_ref', input = 'int'),
          virtual('insert_record_event', returns='void', p = 'program*', ins ='instruction_ref', input = 'int'),
          virtual('insert_wait_event', returns='void', p = 'program*', ins = 'instruction_ref', input = 'int'),
          virtual('insert_stream', returns='void', p = 'program*', ins ='instruction_ref', input = 'int')
)
%>

#endif
} // namespace migraphx

#endif
