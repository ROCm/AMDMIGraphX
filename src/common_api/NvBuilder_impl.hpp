#ifndef NV_BUILDER_IMPL_H
#define NV_BUILDER_IMPL_H

#include <memory>
#include "migraphx/common_api/NvInfer.h"

namespace nvinfer1
{
    class NvBuilder_impl : public IBuilder, public apiv::VBuilder
    {
    public:
        NvBuilder_impl(void* logger, int32_t version) noexcept;
        ~NvBuilder_impl() override;

        // public API
        bool platformHasFastFp16() const noexcept override;
        bool platformHasFastInt8() const noexcept override;
        int32_t getMaxDLABatchSize() const noexcept override;
        int32_t getNbDLACores() const noexcept override;
        void setGpuAllocator(IGpuAllocator* allocator) noexcept override;
        nvinfer1::IBuilderConfig* createBuilderConfig() noexcept override;
        nvinfer1::INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept override;
        nvinfer1::IOptimizationProfile* createOptimizationProfile() noexcept override;
        void setErrorRecorder(IErrorRecorder* recorder) noexcept override;
        IErrorRecorder* getErrorRecorder() const noexcept override;
        void reset() noexcept override;
        bool platformHasTf32() const noexcept override;
        nvinfer1::IHostMemory* buildSerializedNetwork(
            INetworkDefinition& network, IBuilderConfig& config) noexcept override;
        bool isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept override;
        ILogger* getLogger() const noexcept override;
        bool setMaxThreads(int32_t maxThreads) noexcept override;
        int32_t getMaxThreads() const noexcept override;
        IPluginRegistry& getPluginRegistry() noexcept override;
        ICudaEngine* buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept override;
        bool buildSerializedNetworkToStream(
            INetworkDefinition& network, IBuilderConfig& config, IStreamWriter& writer) noexcept override;
        nvinfer1::IHostMemory* buildSerializedNetworkWithKernelText(
            INetworkDefinition& network, IBuilderConfig& config, IHostMemory*& kernelText) noexcept override;

    private:
        IPluginRegistry* mPluginRegistry;
        std::unique_ptr<INetworkDefinition> mNetworkDefinition;
    };

}  // ns:nvinfer1

#endif // NV_BUILDER_IMPL_H
