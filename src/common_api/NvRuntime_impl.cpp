#include "NvRuntime_impl.hpp"

namespace nvinfer1
{

NvRuntime_impl::NvRuntime_impl(void* logger, int32_t version) noexcept
{
    // TODO! implement
    mImpl = this;
}

NvRuntime_impl::~NvRuntime_impl()
{
    // TODO! implement
}

// public API
IRuntime* NvRuntime_impl::getPImpl() noexcept
{
    // TODO! implement
    return nullptr;
}

nvinfer1::ICudaEngine* NvRuntime_impl::deserializeCudaEngine(void const* blob, std::size_t size) noexcept
{
    // TODO! implement
    return nullptr;
}

nvinfer1::ICudaEngine* NvRuntime_impl::deserializeCudaEngine(IStreamReader& streamReader) noexcept
{
    // TODO! implement
    return nullptr;
}

void NvRuntime_impl::setDLACore(int32_t dlaCore) noexcept
{
    // TODO! implement
}

int32_t NvRuntime_impl::getDLACore() const noexcept
{
    // TODO! implement
    return 0;
}

int32_t NvRuntime_impl::getNbDLACores() const noexcept
{
    // TODO! implement
    return 0;
}

void NvRuntime_impl::setGpuAllocator(IGpuAllocator* allocator) noexcept
{
    // TODO! implement
}

void NvRuntime_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
    // TODO! implement
}

IErrorRecorder* NvRuntime_impl::getErrorRecorder() const noexcept
{
    // TODO! implement
    return nullptr;
}

ILogger* NvRuntime_impl::getLogger() const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvRuntime_impl::setMaxThreads(int32_t maxThreads) noexcept
{
    // TODO! implement
    return false;
}

int32_t NvRuntime_impl::getMaxThreads() const noexcept
{
    // TODO! implement
    return 0;
}

void NvRuntime_impl::setTemporaryDirectory(char const*) noexcept
{
    // TODO! implement
}

char const* NvRuntime_impl::getTemporaryDirectory() const noexcept
{
    // TODO! implement
    return nullptr;
}

void NvRuntime_impl::setTempfileControlFlags(TempfileControlFlags) noexcept
{
    // TODO! implement
}

TempfileControlFlags NvRuntime_impl::getTempfileControlFlags() const noexcept
{
    // TODO! implement
    return 0;
}

IPluginRegistry& NvRuntime_impl::getPluginRegistry() noexcept
{
    // TODO! implement
    return *reinterpret_cast<IPluginRegistry*>(mPluginRegistry);
}

void NvRuntime_impl::setPluginRegistryParent(IPluginRegistry* parent) noexcept
{
    // TODO! implement
}

IRuntime* NvRuntime_impl::loadRuntime(char const* path) noexcept
{
    // TODO! implement
    return nullptr;
}

void NvRuntime_impl::setEngineHostCodeAllowed(bool allowed) noexcept
{
    // TODO! implement
}

bool NvRuntime_impl::getEngineHostCodeAllowed() const noexcept
{
    // TODO! implement
    return false;
}

// Added in TensorRT version 10.7
nvinfer1::ICudaEngine* NvRuntime_impl::deserializeCudaEngineV2(IStreamReaderV2& streamReader) noexcept
{
    // TODO! implement
    return nullptr;
}

}   // ns:nvinfer1
