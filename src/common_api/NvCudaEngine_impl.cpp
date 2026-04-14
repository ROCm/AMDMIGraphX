#include "migraphx/common_api/NvInfer.h"
#include "NvCudaEngine_impl.hpp"
#include "NvExecutionContext_impl.hpp"

namespace nvinfer1
{

NvCudaEngine_impl::NvCudaEngine_impl(void* logger, int32_t version) noexcept
{
    // TODO! implement
    mImpl = this;
}

NvCudaEngine_impl::NvCudaEngine_impl(const std::shared_ptr<migraphx::program>& program) noexcept
    : mProgram(program), mTensorNames{program->get_parameter_names()}  
{
    // TODO! implement
    mImpl = this;
}

NvCudaEngine_impl::~NvCudaEngine_impl()
{
    // TODO! implement
}

ICudaEngine* NvCudaEngine_impl::getPImpl() noexcept
{
    // TODO! implement
    return this;
}

int32_t NvCudaEngine_impl::getNbLayers() const noexcept
{
    // TODO! implement
    return 0;
}

IHostMemory* NvCudaEngine_impl::serialize() const noexcept
{
    // TODO! implement
    return nullptr;
}

IExecutionContext* NvCudaEngine_impl::createExecutionContext(ExecutionContextAllocationStrategy strategy) noexcept
{
    return new NvExecutionContext_impl{mProgram};
}

IExecutionContext* NvCudaEngine_impl::createExecutionContextWithoutDeviceMemory() noexcept
{
    // TODO! implement
    return nullptr;
}

size_t NvCudaEngine_impl::getDeviceMemorySize() const noexcept
{
    // TODO! implement
    return 0;
}

bool NvCudaEngine_impl::isRefittable() const noexcept
{
    // TODO! implement
    return false;
}

char const* NvCudaEngine_impl::getName() const noexcept
{
    // TODO! implement
    return nullptr;
}

int32_t NvCudaEngine_impl::getNbOptimizationProfiles() const noexcept
{
    // TODO! implement
    return 0;
}

int32_t const* NvCudaEngine_impl::getProfileTensorValues(
    char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
{
    // TODO! implement
    return nullptr;
}

EngineCapability NvCudaEngine_impl::getEngineCapability() const noexcept
{
    // TODO! implement
    return EngineCapability::kSTANDARD;
}

void NvCudaEngine_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
    // TODO! implement
}

IErrorRecorder* NvCudaEngine_impl::getErrorRecorder() const noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvCudaEngine_impl::hasImplicitBatchDimension() const noexcept
{
    // TODO! implement
    return false;
}

TacticSources NvCudaEngine_impl::getTacticSources() const noexcept
{
    // TODO! implement
    return 0;
}

ProfilingVerbosity NvCudaEngine_impl::getProfilingVerbosity() const noexcept
{
    // TODO! implement
    return ProfilingVerbosity::kLAYER_NAMES_ONLY;
}

IEngineInspector* NvCudaEngine_impl::createEngineInspector() const noexcept
{
    // TODO! implement
    return nullptr;
}

Dims NvCudaEngine_impl::getTensorShape(char const* tensorName) const noexcept
{
    // TODO! implement
    return Dims{};
}

DataType NvCudaEngine_impl::getTensorDataType(char const* tensorName) const noexcept
{
    // TODO! implement
    return DataType::kFLOAT;
}

TensorLocation NvCudaEngine_impl::getTensorLocation(char const* tensorName) const noexcept
{   
    // TODO! implement
    return TensorLocation::kDEVICE;
}

bool NvCudaEngine_impl::isShapeInferenceIO(char const* tensorName) const noexcept
{
    // TODO! implement
    return false;
}

TensorIOMode NvCudaEngine_impl::getTensorIOMode(char const* tensorName) const noexcept
{
    // TODO! implement
    return TensorIOMode::kNONE;
}

int32_t NvCudaEngine_impl::getTensorBytesPerComponent(char const* tensorName) const noexcept
{
    // TODO! implement
    return 0;
}

int32_t NvCudaEngine_impl::getTensorComponentsPerElement(char const* tensorName) const noexcept
{
    // TODO! implement
    return 0;
}

TensorFormat NvCudaEngine_impl::getTensorFormat(char const* tensorName) const noexcept
{
    // TODO! implement
    return TensorFormat::kLINEAR;
}

char const* NvCudaEngine_impl::getTensorFormatDesc(char const* tensorName) const noexcept
{
    // TODO! implement
    return nullptr;
}

int32_t NvCudaEngine_impl::getTensorVectorizedDim(char const* tensorName) const noexcept
{
    // TODO! implement
    return -1;
}

Dims NvCudaEngine_impl::getProfileShape(
    char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
{
    // TODO! implement
    return Dims{};
}

int32_t NvCudaEngine_impl::getNbIOTensors() const noexcept
{
    // TODO! implement
    return 0;
}

char const* NvCudaEngine_impl::getIOTensorName(int32_t index) const noexcept
{
    // TODO! implement
    return nullptr;
}

HardwareCompatibilityLevel NvCudaEngine_impl::getHardwareCompatibilityLevel() const noexcept
{
    // TODO! implement
    return HardwareCompatibilityLevel::kNONE;
}

int32_t NvCudaEngine_impl::getNbAuxStreams() const noexcept
{
    // TODO! implement
    return 0;
}

int32_t NvCudaEngine_impl::getTensorBytesPerComponentV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    // TODO! implement
    return 0;
}

int32_t NvCudaEngine_impl::getTensorComponentsPerElementV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    // TODO! implement
    return 0;
}

TensorFormat NvCudaEngine_impl::getTensorFormatV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    // TODO! implement
    return TensorFormat::kLINEAR;
}

char const* NvCudaEngine_impl::getTensorFormatDescV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    // TODO! implement
    return nullptr;
}

int32_t NvCudaEngine_impl::getTensorVectorizedDimV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    // TODO! implement
    return -1;
}

ISerializationConfig* NvCudaEngine_impl::createSerializationConfig() noexcept
{
    // TODO! implement
    return nullptr;
}

IHostMemory* NvCudaEngine_impl::serializeWithConfig(ISerializationConfig& config) const noexcept
{
    // TODO! implement
    return nullptr;
}

size_t NvCudaEngine_impl::getDeviceMemorySizeForProfile(int32_t profileIndex) const noexcept
{
    // TODO! implement
    return 0;
}

IRefitter* NvCudaEngine_impl::createRefitter(ILogger& logger) noexcept
{
    // TODO! implement
    return nullptr;
}

bool NvCudaEngine_impl::setWeightStreamingBudget(int64_t gpuMemoryBudget) noexcept
{
    // TODO! implement
    return false;
}

int64_t NvCudaEngine_impl::getWeightStreamingBudget() const noexcept
{
    // TODO! implement
    return 0;
}

int64_t NvCudaEngine_impl::getMinimumWeightStreamingBudget() const noexcept
{
    // TODO! implement
    return 0;
}

int64_t NvCudaEngine_impl::getStreamableWeightsSize() const noexcept
{
    // TODO! implement
    return 0;
}

bool NvCudaEngine_impl::isDebugTensor(char const* name) const noexcept
{
    // TODO! implement
    return false;
}

// Added in TensorRT 10.1
bool NvCudaEngine_impl::setWeightStreamingBudgetV2(int64_t gpuMemoryBudget) noexcept
{
    // TODO! implement
    return false;
}

int64_t NvCudaEngine_impl::getWeightStreamingBudgetV2() const noexcept
{
    // TODO! implement
    return 0;
}

int64_t NvCudaEngine_impl::getWeightStreamingAutomaticBudget() const noexcept
{
    // TODO! implement
    return 0;
}

int64_t NvCudaEngine_impl::getWeightStreamingScratchMemorySize() const noexcept
{
    // TODO! implement
    return 0;
}

int64_t NvCudaEngine_impl::getDeviceMemorySizeV2() const noexcept
{
    // TODO! implement
    return 0;
}

int64_t NvCudaEngine_impl::getDeviceMemorySizeForProfileV2(int32_t profileIndex) const noexcept
{
    // TODO! implement
    return 0;
}

// Added in TensorRT 10.11
TRT_NODISCARD int64_t const* NvCudaEngine_impl::getProfileTensorValuesV2(
    char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
{
    // TODO! implement
    return nullptr;
}

TRT_NODISCARD IExecutionContext* NvCudaEngine_impl::createExecutionContextWithRuntimeConfig(
    IRuntimeConfig* runtimeConfig) noexcept
{
    // TODO! implement
    return nullptr;
}

TRT_NODISCARD IRuntimeConfig* NvCudaEngine_impl::createRuntimeConfig() noexcept
{
    // TODO! implement
    return nullptr; 
}

} // ns:nvinfer1
