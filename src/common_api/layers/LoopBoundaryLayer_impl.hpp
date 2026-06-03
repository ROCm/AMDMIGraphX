#ifndef LOOP_BOUNDARY_LAYER_IMPL_HPP
#define LOOP_BOUNDARY_LAYER_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"
#include "Layer_impl.hpp"

namespace nvinfer1
{
    //! Common base for the four loop boundary layers (trip limit, recurrence,
    //! iterator and loop output). Each boundary layer is owned by, and reports
    //! back to, the ILoop facade that created it.
    //!
    //! Note: this class deliberately does NOT inherit ILoopBoundaryLayer. The
    //! concrete interfaces (ITripLimitLayer, ...) already derive from it, so
    //! deriving it here as well would create an ambiguous second subobject. It
    //! only supplies the apiv::VLoopBoundaryLayer implementation; the derived
    //! constructors wire up the interface-side mLayer/mBoundary/mImpl pointers.
    class LoopBoundaryLayer_impl : public apiv::VLoopBoundaryLayer, virtual public Layer_impl
    {
    public:
        LoopBoundaryLayer_impl();
        explicit LoopBoundaryLayer_impl(ILoop* loop);
        ~LoopBoundaryLayer_impl() override;

        // apiv::VLoopBoundaryLayer
        ILoop* getLoop() const noexcept override;

        // No-op: the building of loop boundary layers is orchestrated by the
        // owning ILoop (see Loop_impl::preBuild()/finalize()).
        void build() noexcept override;

    protected:
        ILoop* mLoop = nullptr;
    };

}  // namespace nvinfer1

#endif // LOOP_BOUNDARY_LAYER_IMPL_HPP
