// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "NvExecutionContext_impl.hpp"

namespace nvinfer1
{

NvExecutionContext_impl::NvExecutionContext_impl(void* logger, int32_t version) noexcept
    : mEngine(nullptr)
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

NvExecutionContext_impl::NvExecutionContext_impl(const std::shared_ptr<migraphx::program>& program) noexcept
    : mProgram(program) 
{
    mImpl = this;
}

NvExecutionContext_impl::~NvExecutionContext_impl()
{
    pass_warning("TODO! implement me!", false);
}

IExecutionContext* NvExecutionContext_impl::getPImpl() noexcept
{
    pass_warning("TODO! implement me!", true);
    return this;
}

void NvExecutionContext_impl::setDebugSync(bool sync) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool NvExecutionContext_impl::getDebugSync() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void NvExecutionContext_impl::setProfiler(IProfiler*) noexcept
{
    pass_warning("TODO! implement me!", true);
}

IProfiler* NvExecutionContext_impl::getProfiler() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

ICudaEngine const& NvExecutionContext_impl::getEngine() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return *mEngine;
}

void NvExecutionContext_impl::setName(char const* name) noexcept
{
    pass_warning("TODO! implement me!", true);
}

char const* NvExecutionContext_impl::getName() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvExecutionContext_impl::setDeviceMemory(void* memory) noexcept
{
    pass_warning("TODO! implement me!", true);
}

int32_t NvExecutionContext_impl::getOptimizationProfile() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvExecutionContext_impl::allInputDimensionsSpecified() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::allInputShapesSpecified() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void NvExecutionContext_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
    pass_warning("TODO! implement me!", true);
}

IErrorRecorder* NvExecutionContext_impl::getErrorRecorder() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvExecutionContext_impl::executeV2(void* const* bindings) noexcept
{
    auto result = mProgram->eval(mParamMap);
    return true;
}

bool NvExecutionContext_impl::setOptimizationProfileAsync(int32_t profileIndex, hipStream_t stream) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void NvExecutionContext_impl::setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool NvExecutionContext_impl::getEnqueueEmitsProfile() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::reportToProfiler() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::setInputShape(char const* tensorName, Dims const& dims) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

Dims NvExecutionContext_impl::getTensorShape(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return Dims{};
}

Dims NvExecutionContext_impl::getTensorStrides(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return Dims{};
}

bool NvExecutionContext_impl::setTensorAddress(char const* tensorName, void* data) noexcept
{
    mParamMap[tensorName] = migraphx::argument(mProgram->get_parameter_shapes().at(tensorName), data);
    return true;
}

void const* NvExecutionContext_impl::getTensorAddress(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvExecutionContext_impl::setInputTensorAddress(char const* tensorName, void const* data) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::setOutputTensorAddress(char const* tensorName, void* data) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int32_t NvExecutionContext_impl::inferShapes(int32_t nbMaxNames, char const** tensorNames) noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvExecutionContext_impl::setInputConsumedEvent(hipEvent_t event) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

hipEvent_t NvExecutionContext_impl::getInputConsumedEvent() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void* NvExecutionContext_impl::getOutputTensorAddress(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvExecutionContext_impl::setOutputAllocator(char const* tensorName, IOutputAllocator* outputAllocator) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

IOutputAllocator* NvExecutionContext_impl::getOutputAllocator(char const* name) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

int64_t NvExecutionContext_impl::getMaxOutputSize(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvExecutionContext_impl::setTemporaryStorageAllocator(IGpuAllocator* allocator) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

IGpuAllocator* NvExecutionContext_impl::getTemporaryStorageAllocator() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvExecutionContext_impl::enqueueV3(hipStream_t stream) noexcept
{
    // Asynchronous counterpart of executeV2(). Where executeV2() calls
    // program::eval() with a default (synchronous) execution environment,
    // enqueueV3() must only *enqueue* the work on the caller's HIP stream and
    // return without blocking, so the host can queue follow-up work (e.g. the
    // device->host output copy in the sample) on the same stream.
    //
    // migraphx expresses this through execution_environment{queue, async=true}:
    //   - queue: an any_ptr wrapping the hipStream_t. It must be typed so the
    //     gpu context's queue.get<hipStream_t>() type-check matches; the
    //     templated any_ptr ctor records exactly that type name.
    //   - async=true: migraphx makes its own stream wait_for() the caller's
    //     stream before launching and, when done, has the caller's stream
    //     finish_on() migraphx's completion event. The kernels are launched but
    //     not waited on here, preserving asynchronous semantics.
    migraphx::execution_environment exec_env{migraphx::any_ptr{stream}, true};
    mProgram->eval(mParamMap, exec_env);
    return true;
}

void NvExecutionContext_impl::setPersistentCacheLimit(size_t size) noexcept
{
    pass_warning("TODO! implement me!", true);
}

size_t NvExecutionContext_impl::getPersistentCacheLimit() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvExecutionContext_impl::setNvtxVerbosity(ProfilingVerbosity verbosity) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

ProfilingVerbosity NvExecutionContext_impl::getNvtxVerbosity() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return ProfilingVerbosity::kLAYER_NAMES_ONLY;
}

void NvExecutionContext_impl::setAuxStreams(hipStream_t* auxStreams, int32_t nbStreams) noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool NvExecutionContext_impl::setDebugListener(IDebugListener* listener) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

IDebugListener* NvExecutionContext_impl::getDebugListener() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}   

bool NvExecutionContext_impl::setTensorDebugState(char const* name, bool flag) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::getDebugState(char const* name) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::setAllTensorsDebugState(bool flag) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

size_t NvExecutionContext_impl::updateDeviceMemorySizeForShapes() noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void NvExecutionContext_impl::setDeviceMemoryV2(void* memory, int64_t size) noexcept
{
    pass_warning("TODO! implement me!", true);
}

TRT_NODISCARD IRuntimeConfig* NvExecutionContext_impl::getRuntimeConfig() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvExecutionContext_impl::setUnfusedTensorsDebugState(bool flag) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvExecutionContext_impl::getUnfusedTensorsDebugState() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

}   // ns:nvinfer1
