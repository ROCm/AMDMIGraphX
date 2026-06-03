#ifndef LOOP_OUTPUT_LAYER_IMPL_HPP
#define LOOP_OUTPUT_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "LoopBoundaryLayer_impl.hpp"

namespace nvinfer1
{
    //! The sole way to get a value out of a loop.
    //!  - kLAST_VALUE: output is the final value of input 0 (a recurrence).
    //!  - kCONCATENATE: output stacks input 0 across iterations; input 1 (set via
    //!    setInput(1,...)) gives the concatenation length.
    class LoopOutputLayer_impl : public ILoopOutputLayer, public apiv::VLoopOutputLayer, virtual public LoopBoundaryLayer_impl
    {
    public:
        LoopOutputLayer_impl(ITensor& tensor,
                             LoopOutput outputKind,
                             int32_t axis,
                             const std::shared_ptr<migraphx::program>& program,
                             ILoop* loop) noexcept;
        ~LoopOutputLayer_impl() override;

        // public API
        LoopOutput getLoopOutput() const noexcept override;
        void setAxis(int32_t axis) noexcept override;
        int32_t getAxis() const noexcept override;

        void setInput(int32_t index, ITensor& tensor) noexcept override;

    private:
        LoopOutput mKind;
        int32_t mAxis;
    };

}  // namespace nvinfer1

#endif // LOOP_OUTPUT_LAYER_IMPL_HPP
