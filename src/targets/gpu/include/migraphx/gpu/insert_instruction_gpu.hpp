#ifndef MIGRAPHX_GUARD_RTGLIB_INSERT_INSTRUCTION_GPU_HPP
#define MIGRAPHX_GUARD_RTGLIB_INSERT_INSTRUCTION_GPU_HPP

#include <migraphx/gpu/event.hpp>
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct insert_instruction_gpu
{
    void insert_create_events(program* p, instruction_ref ins, int num_of_events)
    {
        p->insert_instruction(ins, create_events{num_of_events});
    }
    void insert_record_event(program* p, instruction_ref ins, int event)
    {
        p->insert_instruction(ins, record_event{event});
    }
    void insert_wait_event(program* p, instruction_ref ins, int event)
    {
        p->insert_instruction(ins, wait_event{event});
    }
    void insert_stream(program* p, instruction_ref ins, int stream)
    {

        p->insert_instruction(ins, set_stream{stream});
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
