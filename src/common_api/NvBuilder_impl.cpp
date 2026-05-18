// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/register_target.hpp>
#include <migraphx/load_save.hpp>

#include "migraphx/common_api/NvInferImpl.h"
#include "migraphx/common_api/NvInferRuntime.h"

#include "NvBuilder_impl.hpp"
#include "NvHostMemory_impl.hpp"
#include "NetworkDefinition_impl.hpp"
#include "NvBuilderConfig_impl.hpp"

namespace nvinfer1
{

NvBuilder_impl::NvBuilder_impl(void* logger, int32_t version) noexcept
    :  mPluginRegistry(nullptr), mNetworkDefinition(nullptr)
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

NvBuilder_impl::~NvBuilder_impl()
{
    pass_warning("TODO! implement me!", false);
}

bool NvBuilder_impl::platformHasFastFp16() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return true;
}

bool NvBuilder_impl::platformHasFastInt8() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return true;
}

int32_t NvBuilder_impl::getMaxDLABatchSize() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t NvBuilder_impl::getNbDLACores() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void NvBuilder_impl::setGpuAllocator(IGpuAllocator* allocator) noexcept
{
    pass_warning("TODO! implement me!", true);
}

nvinfer1::IBuilderConfig* NvBuilder_impl::createBuilderConfig() noexcept
{
    mBuilderConfig = std::make_unique<NvBuilderConfig_impl>();
    return mBuilderConfig.release();
}

nvinfer1::INetworkDefinition* NvBuilder_impl::createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept
{
    mNetworkDefinition = std::make_unique<NvNetworkDefinition_impl>(flags, *this);
    return mNetworkDefinition.release();
}

nvinfer1::IOptimizationProfile* NvBuilder_impl::createOptimizationProfile() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvBuilder_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{    
}

IErrorRecorder* NvBuilder_impl::getErrorRecorder() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvBuilder_impl::reset() noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool NvBuilder_impl::platformHasTf32() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

nvinfer1::IHostMemory* NvBuilder_impl::buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept
{
    auto& nw_impl = static_cast<NvNetworkDefinition_impl&>(network);
    
    nw_impl.build();
    
    migraphx::program prog = *nw_impl.getProgram();
    try
    {
        prog.compile(migraphx::make_target("gpu"));
    }
    catch(migraphx::exception& /*e*/)
    {
        // TODO write to error recorder/logger
        return nullptr;
    }
    
    mSerializedNetworks.push_back(migraphx::save_buffer(prog));
    auto&& current_network = mSerializedNetworks.back();

    return new NvHostMemory_impl{reinterpret_cast<void*>(current_network.data()),
                            current_network.size(),
                            DataType::kUINT8};
}

bool NvBuilder_impl::isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

ILogger* NvBuilder_impl::getLogger() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvBuilder_impl::setMaxThreads(int32_t maxThreads) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int32_t NvBuilder_impl::getMaxThreads() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

IPluginRegistry& NvBuilder_impl::getPluginRegistry() noexcept
{
    pass_warning("TODO! implement me!", true);
    return *reinterpret_cast<IPluginRegistry*>(mPluginRegistry);
}

ICudaEngine* NvBuilder_impl::buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvBuilder_impl::buildSerializedNetworkToStream(
    INetworkDefinition& network, IBuilderConfig& config, IStreamWriter& writer) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

nvinfer1::IHostMemory* NvBuilder_impl::buildSerializedNetworkWithKernelText(
    INetworkDefinition& network, IBuilderConfig& config, IHostMemory*& kernelText) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

}  // ns:nvinfer1
