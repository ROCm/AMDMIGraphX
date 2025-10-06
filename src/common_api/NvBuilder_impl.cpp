#include "migraphx/common_api/NvInferImpl.h"
#include "migraphx/common_api/NvInferRuntime.h"

#include "NvBuilder_impl.hpp"
#include "NetworkDefinition_impl.hpp"

namespace nvinfer1
{

NvBuilder_impl::NvBuilder_impl(void* logger, int32_t version) noexcept
    :  mPluginRegistry(nullptr), mNetworkDefinition(nullptr)
{
    // TODO! implement
    mImpl = this;
}

NvBuilder_impl::~NvBuilder_impl()
{
    // TODO! implement
}

bool NvBuilder_impl::platformHasFastFp16() const noexcept
{
    // TODO! implement
    return true;
}

bool NvBuilder_impl::platformHasFastInt8() const noexcept
{
    // TODO! implement
    return true;
}

int32_t NvBuilder_impl::getMaxDLABatchSize() const noexcept
{
    // TODO! implement
    return 0;
}

int32_t NvBuilder_impl::getNbDLACores() const noexcept
{
    // TODO! implement
    return 0;
}

void NvBuilder_impl::setGpuAllocator(IGpuAllocator* allocator) noexcept
{
    // TODO! implement
}

nvinfer1::IBuilderConfig* NvBuilder_impl::createBuilderConfig() noexcept
{
    // TODO! implement
    return nullptr;
}

nvinfer1::INetworkDefinition* NvBuilder_impl::createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept
{
    // TODO! implement
    /*
    static NvNetworkDefinition_impl networkDefinition(flags);
    return &networkDefinition;
    */
    mNetworkDefinition = std::make_unique<NvNetworkDefinition_impl>(flags, *this);
    return mNetworkDefinition.get();
}

nvinfer1::IOptimizationProfile* NvBuilder_impl::createOptimizationProfile() noexcept
{
    // TODO! implement
    return nullptr;
}

void NvBuilder_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{    
}

IErrorRecorder* NvBuilder_impl::getErrorRecorder() const noexcept
{
    // TODO! implement
    return nullptr;
}

void NvBuilder_impl::reset() noexcept
{
    // TODO! implement
}

bool NvBuilder_impl::platformHasTf32() const noexcept
{
    // TODO! implement
    return false;
}

nvinfer1::IHostMemory* NvBuilder_impl::buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvBuilder_impl::isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept
{
    // TODO! implement
    return false;
}

ILogger* NvBuilder_impl::getLogger() const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvBuilder_impl::setMaxThreads(int32_t maxThreads) noexcept
{
    // TODO! implement
    return false;
}

int32_t NvBuilder_impl::getMaxThreads() const noexcept
{
    // TODO! implement
    return 0;
}

IPluginRegistry& NvBuilder_impl::getPluginRegistry() noexcept
{
    // TODO! implement
    return *reinterpret_cast<IPluginRegistry*>(mPluginRegistry);
}

ICudaEngine* NvBuilder_impl::buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvBuilder_impl::buildSerializedNetworkToStream(
    INetworkDefinition& network, IBuilderConfig& config, IStreamWriter& writer) noexcept
{
    // TODO! implement
    return false;
}

nvinfer1::IHostMemory* NvBuilder_impl::buildSerializedNetworkWithKernelText(
    INetworkDefinition& network, IBuilderConfig& config, IHostMemory*& kernelText) noexcept
{
    // TODO! implement
    return nullptr;
}

}  // ns:nvinfer1
