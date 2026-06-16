#ifndef LOOP_IMPL_HPP
#define LOOP_IMPL_HPP

#include <memory>
#include <string>
#include <vector>

#include <migraphx/program.hpp>
#include <migraphx/instruction_ref.hpp>

#include "migraphx/common_api/NvInfer.h"

namespace nvinfer1
{
    class NvNetworkDefinition_impl;
    class Layer_impl;
    class Tensor_impl;
    class TripLimitLayer_impl;
    class RecurrenceLayer_impl;
    class IteratorLayer_impl;
    class LoopOutputLayer_impl;

    //! Facade that adapts TensorRT's incremental loop-building API onto a
    //! migraphx "loop" instruction with a body submodule.
    //!
    //! The boundary layers (trip limit, recurrence, iterator, loop output) are
    //! created and owned here. The actual migraphx graph is assembled in two
    //! phases that bracket the building of the regular network layers:
    //!   preBuild()  -> create the submodule + parameters, bind recurrence /
    //!                  iterator outputs, and redirect body layers into it.
    //!   finalize()  -> emit the loop instruction and bind the loop outputs.
    class Loop_impl : public ILoop, public apiv::VLoop
    {
    public:
        Loop_impl(const std::shared_ptr<migraphx::program>& program,
                  NvNetworkDefinition_impl* network,
                  int index);
        ~Loop_impl() override;

        // public API
        IRecurrenceLayer* addRecurrence(ITensor& initialValue) noexcept override;
        ITripLimitLayer* addTripLimit(ITensor& tensor, TripLimit limit) noexcept override;
        IIteratorLayer* addIterator(ITensor& tensor, int32_t axis = 0, bool reverse = false) noexcept override;
        ILoopOutputLayer* addLoopOutput(ITensor& tensor, LoopOutput outputKind, int32_t axis = 0) noexcept override;
        void setName(char const* name) noexcept override;
        char const* getName() const noexcept override;

        // build orchestration (invoked by NvNetworkDefinition_impl::build())
        //
        // The orchestrator drives a dependency-ordered fixpoint instead of a
        // rigid set of phases, so chained loops (loop N feeding loop N+1) build
        // in the right order:
        //   assignModules() - create the body submodule and assign body layers
        //                     to it. Pointer-only; safe before any instruction
        //                     exists.
        //   preBuild()      - once all boundary inputs are bound, create the
        //                     submodule parameters and bind recurrence/iterator
        //                     outputs so the body layers become buildable.
        //   finalize()      - once the body back-edges and scan outputs are
        //                     bound, emit the loop instruction and bind outputs.
        void assignModules() noexcept;
        void preBuild() noexcept;
        void finalize() noexcept;

        // True when every input that feeds a loop boundary (recurrence initial
        // value, iterator data, trip count) has been bound, i.e. preBuild() may
        // run.
        bool preBuildReady() const noexcept;
        // True when every value the body return depends on (recurrence
        // back-edges and scan/last-value sources) has been bound, i.e.
        // finalize() may run.
        bool finalizeReady() const noexcept;

        bool isPreBuilt() const noexcept { return mPreBuilt; }
        bool isFinalized() const noexcept { return mFinalized; }
        const migraphx::module* body() const noexcept { return mBody; }

    private:
        void markBodyLayers() noexcept;
        bool isOutputConsumed(const Tensor_impl* tensor) const noexcept;
        int64_t readTripCount() const noexcept;
        int recurrenceIndex(const Tensor_impl* tensor) const noexcept;

        std::shared_ptr<migraphx::program> mProgram;
        NvNetworkDefinition_impl* mNetwork;
        int mIndex;
        std::string mName;

        migraphx::module* mBody = nullptr;
        migraphx::instruction_ref mIterParam;
        migraphx::instruction_ref mCondParam;
        bool mPreBuilt  = false;
        bool mFinalized = false;

        std::vector<std::unique_ptr<Layer_impl>> mOwned;
        std::vector<TripLimitLayer_impl*> mTripLimits;
        std::vector<RecurrenceLayer_impl*> mRecurrences;
        std::vector<IteratorLayer_impl*> mIterators;
        std::vector<LoopOutputLayer_impl*> mLoopOutputs;
    };

}  // ns:nvinfer1

#endif // LOOP_IMPL_HPP
