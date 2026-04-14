#include "NvExecutionContext_impl.hpp"

namespace nvinfer1
{

NvExecutionContext_impl::NvExecutionContext_impl(void* logger, int32_t version) noexcept
    : mEngine(nullptr)
{
    // TODO! implement
    mImpl = this;
}

NvExecutionContext_impl::NvExecutionContext_impl(const std::shared_ptr<migraphx::program>& program) noexcept
    : mProgram(program) 
{
    mImpl = this;
}

NvExecutionContext_impl::~NvExecutionContext_impl()
{
    // TODO! implement
}

IExecutionContext* NvExecutionContext_impl::getPImpl() noexcept
{
    // TODO! implement
    return this;
}

void NvExecutionContext_impl::setDebugSync(bool sync) noexcept
{
    // TODO! implement
}

bool NvExecutionContext_impl::getDebugSync() const noexcept
{
    // TODO! implement
    return false;
}

void NvExecutionContext_impl::setProfiler(IProfiler*) noexcept
{
    // TODO! implement
}

IProfiler* NvExecutionContext_impl::getProfiler() const noexcept
{
    // TODO! implement
    return nullptr;
}

ICudaEngine const& NvExecutionContext_impl::getEngine() const noexcept
{
    // TODO! implement
    return *mEngine;
}

void NvExecutionContext_impl::setName(char const* name) noexcept
{
    // TODO! implement
}

char const* NvExecutionContext_impl::getName() const noexcept
{
    // TODO! implement
    return nullptr;
}

void NvExecutionContext_impl::setDeviceMemory(void* memory) noexcept
{
    // TODO! implement
}

int32_t NvExecutionContext_impl::getOptimizationProfile() const noexcept
{
    // TODO! implement
    return 0;
}

bool NvExecutionContext_impl::allInputDimensionsSpecified() const noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::allInputShapesSpecified() const noexcept
{
    // TODO! implement
    return false;
}

void NvExecutionContext_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
    // TODO! implement
}

IErrorRecorder* NvExecutionContext_impl::getErrorRecorder() const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvExecutionContext_impl::executeV2(void* const* bindings) noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::setOptimizationProfileAsync(int32_t profileIndex, hipStream_t stream) noexcept
{
    // TODO! implement
    return false;
}

void NvExecutionContext_impl::setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept
{
    // TODO! implement
}

bool NvExecutionContext_impl::getEnqueueEmitsProfile() const noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::reportToProfiler() const noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::setInputShape(char const* tensorName, Dims const& dims) noexcept
{
    // TODO! implement
    return false;
}

Dims NvExecutionContext_impl::getTensorShape(char const* tensorName) const noexcept
{
    // TODO! implement
    return Dims{};
}

Dims NvExecutionContext_impl::getTensorStrides(char const* tensorName) const noexcept
{
    // TODO! implement
    return Dims{};
}

bool NvExecutionContext_impl::setTensorAddress(char const* tensorName, void* data) noexcept
{
    mParamMap[tensorName] = migraphx::argument(mProgram->get_parameter_shapes().at(tensorName), data);
    return true;
}

void const* NvExecutionContext_impl::getTensorAddress(char const* tensorName) const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvExecutionContext_impl::setInputTensorAddress(char const* tensorName, void const* data) noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::setOutputTensorAddress(char const* tensorName, void* data) noexcept
{
    // TODO! implement
    return false;
}

int32_t NvExecutionContext_impl::inferShapes(int32_t nbMaxNames, char const** tensorNames) noexcept
{
    // TODO! implement
    return 0;
}

bool NvExecutionContext_impl::setInputConsumedEvent(hipEvent_t event) noexcept
{
    // TODO! implement
    return false;
}

hipEvent_t NvExecutionContext_impl::getInputConsumedEvent() const noexcept
{
    // TODO! implement
    return nullptr;
}

void* NvExecutionContext_impl::getOutputTensorAddress(char const* tensorName) const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvExecutionContext_impl::setOutputAllocator(char const* tensorName, IOutputAllocator* outputAllocator) noexcept
{
    // TODO! implement
    return false;
}

IOutputAllocator* NvExecutionContext_impl::getOutputAllocator(char const* name) noexcept
{
    // TODO! implement
    return nullptr;
}

int64_t NvExecutionContext_impl::getMaxOutputSize(char const* tensorName) const noexcept
{
    // TODO! implement
    return 0;
}

bool NvExecutionContext_impl::setTemporaryStorageAllocator(IGpuAllocator* allocator) noexcept
{
    // TODO! implement
    return false;
}

IGpuAllocator* NvExecutionContext_impl::getTemporaryStorageAllocator() const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvExecutionContext_impl::enqueueV3(hipStream_t stream) noexcept
{
    // TODO! implement
    return false;
}

void NvExecutionContext_impl::setPersistentCacheLimit(size_t size) noexcept
{
    // TODO! implement
}

size_t NvExecutionContext_impl::getPersistentCacheLimit() const noexcept
{
    // TODO! implement
    return 0;
}

bool NvExecutionContext_impl::setNvtxVerbosity(ProfilingVerbosity verbosity) noexcept
{
    // TODO! implement
    return false;
}

ProfilingVerbosity NvExecutionContext_impl::getNvtxVerbosity() const noexcept
{
    // TODO! implement
    return ProfilingVerbosity::kLAYER_NAMES_ONLY;
}

void NvExecutionContext_impl::setAuxStreams(hipStream_t* auxStreams, int32_t nbStreams) noexcept
{
    // TODO! implement
}

bool NvExecutionContext_impl::setDebugListener(IDebugListener* listener) noexcept
{
    // TODO! implement
    return false;
}

IDebugListener* NvExecutionContext_impl::getDebugListener() noexcept
{
    // TODO! implement
    return nullptr;
}   

bool NvExecutionContext_impl::setTensorDebugState(char const* name, bool flag) noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::getDebugState(char const* name) const noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::setAllTensorsDebugState(bool flag) noexcept
{
    // TODO! implement
    return false;
}

size_t NvExecutionContext_impl::updateDeviceMemorySizeForShapes() noexcept
{
    // TODO! implement
    return 0;
}

void NvExecutionContext_impl::setDeviceMemoryV2(void* memory, int64_t size) noexcept
{
    // TODO! implement
}

TRT_NODISCARD IRuntimeConfig* NvExecutionContext_impl::getRuntimeConfig() const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvExecutionContext_impl::setUnfusedTensorsDebugState(bool flag) noexcept
{
    // TODO! implement
    return false;
}

bool NvExecutionContext_impl::getUnfusedTensorsDebugState() const noexcept
{
    // TODO! implement
    return false;
}

}   // ns:nvinfer1
