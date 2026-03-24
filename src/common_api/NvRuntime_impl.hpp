#ifndef NV_RUNTIME_IMPL_H
#define NV_RUNTIME_IMPL_H

#include "migraphx/common_api/NvInferRuntime.h"

namespace nvinfer1
{
    class NvRuntime_impl : public IRuntime, public apiv::VRuntime
    {
    public:
        NvRuntime_impl(void* logger, int32_t version) noexcept;
        ~NvRuntime_impl() override;
    
        // public API
        IRuntime* getPImpl() noexcept override;
        nvinfer1::ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size) noexcept override;
        nvinfer1::ICudaEngine* deserializeCudaEngine(IStreamReader& streamReader) noexcept override;
        void setDLACore(int32_t dlaCore) noexcept override;
        int32_t getDLACore() const noexcept override;
        int32_t getNbDLACores() const noexcept override;
        void setGpuAllocator(IGpuAllocator* allocator) noexcept override;
        void setErrorRecorder(IErrorRecorder* recorder) noexcept override;
        IErrorRecorder* getErrorRecorder() const noexcept override;
        ILogger* getLogger() const noexcept override;
        bool setMaxThreads(int32_t maxThreads) noexcept override;
        int32_t getMaxThreads() const noexcept override;
        void setTemporaryDirectory(char const*) noexcept override;
        char const* getTemporaryDirectory() const noexcept override;
        void setTempfileControlFlags(TempfileControlFlags) noexcept override;
        TempfileControlFlags getTempfileControlFlags() const noexcept override;
        IPluginRegistry& getPluginRegistry() noexcept override;
        void setPluginRegistryParent(IPluginRegistry* parent) noexcept override;
        IRuntime* loadRuntime(char const* path) noexcept override;
        void setEngineHostCodeAllowed(bool allowed) noexcept override;
        bool getEngineHostCodeAllowed() const noexcept override;
        // Added in TensorRT version 10.7
        nvinfer1::ICudaEngine* deserializeCudaEngineV2(IStreamReaderV2& streamReader) noexcept override;

    private:
        IPluginRegistry* mPluginRegistry;
    };

}   // ns:nvinfer1

#endif // NV_RUNTIME_IMPL_H