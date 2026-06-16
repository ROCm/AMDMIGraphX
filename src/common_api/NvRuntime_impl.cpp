// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "NvRuntime_impl.hpp"

#include <migraphx/program.hpp>
#include <migraphx/load_save.hpp>

#include "NvCudaEngine_impl.hpp"

namespace nvinfer1
{

NvRuntime_impl::NvRuntime_impl(void* logger, int32_t version) noexcept
{
    mImpl = this;
}

NvRuntime_impl::~NvRuntime_impl()
{
}

// public API
IRuntime* NvRuntime_impl::getPImpl() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

nvinfer1::ICudaEngine* NvRuntime_impl::deserializeCudaEngine(void const* blob, std::size_t size) noexcept
{
    std::shared_ptr<migraphx::program> program;
    try
    {
        program = std::make_shared<migraphx::program>(
            migraphx::load_buffer(reinterpret_cast<const char*>(blob), size));
    }
    catch(migraphx::exception e)
    {
        // TODO write to error recorder if set, otherwise to logger
        return nullptr;
    }

    auto* engine = new NvCudaEngine_impl{std::move(program)};
    /*
    if(error_recorder_)
        engine->setErrorRecorder(error_recorder_);
    */

    return engine;
}

nvinfer1::ICudaEngine* NvRuntime_impl::deserializeCudaEngine(IStreamReader& streamReader) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvRuntime_impl::setDLACore(int32_t dlaCore) noexcept
{
    pass_warning("TODO! implement me!", true);
}

int32_t NvRuntime_impl::getDLACore() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t NvRuntime_impl::getNbDLACores() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void NvRuntime_impl::setGpuAllocator(IGpuAllocator* allocator) noexcept
{
    pass_warning("TODO! implement me!", true);
}

void NvRuntime_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
    pass_warning("TODO! implement me!", true);
}

IErrorRecorder* NvRuntime_impl::getErrorRecorder() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

ILogger* NvRuntime_impl::getLogger() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvRuntime_impl::setMaxThreads(int32_t maxThreads) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int32_t NvRuntime_impl::getMaxThreads() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void NvRuntime_impl::setTemporaryDirectory(char const*) noexcept
{
    pass_warning("TODO! implement me!", true);
}

char const* NvRuntime_impl::getTemporaryDirectory() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvRuntime_impl::setTempfileControlFlags(TempfileControlFlags) noexcept
{
    pass_warning("TODO! implement me!", true);
}

TempfileControlFlags NvRuntime_impl::getTempfileControlFlags() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

IPluginRegistry& NvRuntime_impl::getPluginRegistry() noexcept
{
    pass_warning("TODO! implement me!", true);
    return *reinterpret_cast<IPluginRegistry*>(mPluginRegistry);
}

void NvRuntime_impl::setPluginRegistryParent(IPluginRegistry* parent) noexcept
{
    pass_warning("TODO! implement me!", true);
}

IRuntime* NvRuntime_impl::loadRuntime(char const* path) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvRuntime_impl::setEngineHostCodeAllowed(bool allowed) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool NvRuntime_impl::getEngineHostCodeAllowed() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

// Added in TensorRT version 10.7
nvinfer1::ICudaEngine* NvRuntime_impl::deserializeCudaEngineV2(IStreamReaderV2& streamReader) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

}   // ns:nvinfer1
