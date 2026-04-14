#ifndef NV_EXECUTION_CONTEXT_IMPL_HPP
#define NV_EXECUTION_CONTEXT_IMPL_HPP

#include <memory>
#include <migraphx/program.hpp>
#include "migraphx/common_api/NvInferRuntime.h"

namespace nvinfer1
{
    class NvExecutionContext_impl : public IExecutionContext, public apiv::VExecutionContext
    {
    public:
        NvExecutionContext_impl(void* logger, int32_t version) noexcept;
        NvExecutionContext_impl(const std::shared_ptr<migraphx::program>& program) noexcept;
        ~NvExecutionContext_impl() override;

        // public API
    public:
        IExecutionContext* getPImpl() noexcept override;
        void setDebugSync(bool sync) noexcept override;
        bool getDebugSync() const noexcept override;
        void setProfiler(IProfiler*) noexcept override;
        IProfiler* getProfiler() const noexcept override;
        ICudaEngine const& getEngine() const noexcept override;
        void setName(char const* name) noexcept override;
        char const* getName() const noexcept override;
        void setDeviceMemory(void* memory) noexcept override;
        int32_t getOptimizationProfile() const noexcept override;
        bool allInputDimensionsSpecified() const noexcept override;
        bool allInputShapesSpecified() const noexcept override;
        void setErrorRecorder(IErrorRecorder* recorder) noexcept override;
        IErrorRecorder* getErrorRecorder() const noexcept override;
        bool executeV2(void* const* bindings) noexcept override;
        bool setOptimizationProfileAsync(int32_t profileIndex, hipStream_t stream) noexcept override;
        void setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept override;
        bool getEnqueueEmitsProfile() const noexcept override;
        bool reportToProfiler() const noexcept override;
        bool setInputShape(char const* tensorName, Dims const& dims) noexcept override;
        Dims getTensorShape(char const* tensorName) const noexcept override;
        Dims getTensorStrides(char const* tensorName) const noexcept override;
        bool setTensorAddress(char const* tensorName, void* data) noexcept override;
        void const* getTensorAddress(char const* tensorName) const noexcept override;
        bool setInputTensorAddress(char const* tensorName, void const* data) noexcept override;
        bool setOutputTensorAddress(char const* tensorName, void* data) noexcept override;
        int32_t inferShapes(int32_t nbMaxNames, char const** tensorNames) noexcept override;
        bool setInputConsumedEvent(hipEvent_t event) noexcept override;
        hipEvent_t getInputConsumedEvent() const noexcept override;
        void* getOutputTensorAddress(char const* tensorName) const noexcept override;
        bool setOutputAllocator(char const* tensorName, IOutputAllocator* outputAllocator) noexcept override;
        IOutputAllocator* getOutputAllocator(char const* name) noexcept override;
        int64_t getMaxOutputSize(char const* tensorName) const noexcept override;
        bool setTemporaryStorageAllocator(IGpuAllocator* allocator) noexcept override;
        IGpuAllocator* getTemporaryStorageAllocator() const noexcept override;
        bool enqueueV3(hipStream_t stream) noexcept override;
        void setPersistentCacheLimit(size_t size) noexcept override;
        size_t getPersistentCacheLimit() const noexcept override;
        bool setNvtxVerbosity(ProfilingVerbosity verbosity) noexcept override;
        ProfilingVerbosity getNvtxVerbosity() const noexcept override;
        void setAuxStreams(hipStream_t* auxStreams, int32_t nbStreams) noexcept override;
        bool setDebugListener(IDebugListener* listener) noexcept override;
        IDebugListener* getDebugListener() noexcept override;
        bool setTensorDebugState(char const* name, bool flag) noexcept override;
        bool getDebugState(char const* name) const noexcept override;
        bool setAllTensorsDebugState(bool flag) noexcept override;
        size_t updateDeviceMemorySizeForShapes() noexcept override;
        void setDeviceMemoryV2(void* memory, int64_t size) noexcept override;
        TRT_NODISCARD IRuntimeConfig* getRuntimeConfig() const noexcept override;
        bool setUnfusedTensorsDebugState(bool flag) noexcept override;
        bool getUnfusedTensorsDebugState() const noexcept override;

    private:
        std::unique_ptr<ICudaEngine> mEngine;

        std::shared_ptr<migraphx::program> mProgram;
        migraphx::parameter_map mParamMap;
    };

} // namespace nvinfer1

#endif // NV_EXECUTION_CONTEXT_IMPL_HPP
