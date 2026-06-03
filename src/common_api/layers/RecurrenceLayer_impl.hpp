#ifndef RECURRENCE_LAYER_IMPL_HPP
#define RECURRENCE_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    //! A recurrence defines a loop-carried value:
    //!  - input 0: the initial value (lives outside the loop)
    //!  - input 1: the next value (lives inside the loop body), set via setInput(1, ...)
    //!  - output 0: the value as seen at the start of an iteration (the body parameter)
    class RecurrenceLayer_impl : public IRecurrenceLayer, public apiv::VRecurrenceLayer, virtual public LoopBoundaryLayer_impl
    {
    public:
        RecurrenceLayer_impl(ITensor& initialValue,
                             const std::shared_ptr<migraphx::program>& program,
                             ILoop* loop) noexcept;
        ~RecurrenceLayer_impl() override;

        // The recurrence keeps its inputs as plain references; unlike a normal
        // layer it must not rewrite an instruction argument when setInput(1,...)
        // wires the back-edge, because the next value is produced inside the body.
        void setInput(int32_t index, ITensor& tensor) noexcept override;
    };

}  // namespace nvinfer1

#endif // RECURRENCE_LAYER_IMPL_HPP
