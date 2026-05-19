#ifndef LOOP_IMPL_HPP
#define LOOP_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"

namespace nvinfer1
{
    class Loop_impl : public ILoop, public apiv::VLoop
    {
    public:
        Loop_impl();
        ~Loop_impl() override;    

        // public API
        IRecurrenceLayer* addRecurrence(ITensor& initialValue) noexcept override;
        ITripLimitLayer* addTripLimit(ITensor& tensor, TripLimit limit) noexcept override;
        IIteratorLayer* addIterator(ITensor& tensor, int32_t axis = 0, bool reverse = false) noexcept override;
        ILoopOutputLayer* addLoopOutput(ITensor& tensor, LoopOutput outputKind, int32_t axis = 0) noexcept override;
        void setName(char const* name) noexcept override;
        char const* getName() const noexcept override;
    };

}  // ns:nvinfer1

#endif // LOOP_IMPL_HPP

