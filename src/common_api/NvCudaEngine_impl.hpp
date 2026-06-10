#ifndef NV_CUDA_ENGINE_IMPL_H
#define NV_CUDA_ENGINE_IMPL_H

#include <vector>
#include <set>
#include <string>
#include <migraphx/program.hpp>
#include "migraphx/common_api/NvInferRuntime.h"

namespace nvinfer1
{
    class NvCudaEngine_impl : public ICudaEngine, public apiv::VCudaEngine
    {
    public:
        NvCudaEngine_impl(void* logger, int32_t version) noexcept;
        NvCudaEngine_impl(const std::shared_ptr<migraphx::program>& program) noexcept;

        ~NvCudaEngine_impl() override;

        // public API
        ICudaEngine* getPImpl() noexcept override;
        int32_t getNbLayers() const noexcept override;
        IHostMemory* serialize() const noexcept override;
        IExecutionContext* createExecutionContext(ExecutionContextAllocationStrategy strategy) noexcept override;
        IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept override;
        size_t getDeviceMemorySize() const noexcept override;
        bool isRefittable() const noexcept override;
        char const* getName() const noexcept override;
        int32_t getNbOptimizationProfiles() const noexcept override;
        int32_t const* getProfileTensorValues(
            char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept override;
        EngineCapability getEngineCapability() const noexcept override;
        void setErrorRecorder(IErrorRecorder* recorder) noexcept override;
        IErrorRecorder* getErrorRecorder() const noexcept override;
        bool hasImplicitBatchDimension() const noexcept override;
        TacticSources getTacticSources() const noexcept override;
        ProfilingVerbosity getProfilingVerbosity() const noexcept override;
        IEngineInspector* createEngineInspector() const noexcept override;
        Dims getTensorShape(char const* tensorName) const noexcept override;
        DataType getTensorDataType(char const* tensorName) const noexcept override;
        TensorLocation getTensorLocation(char const* tensorName) const noexcept override;
        bool isShapeInferenceIO(char const* tensorName) const noexcept override;
        TensorIOMode getTensorIOMode(char const* tensorName) const noexcept override;
        int32_t getTensorBytesPerComponent(char const* tensorName) const noexcept override;
        int32_t getTensorComponentsPerElement(char const* tensorName) const noexcept override;
        TensorFormat getTensorFormat(char const* tensorName) const noexcept override;
        char const* getTensorFormatDesc(char const* tensorName) const noexcept override;
        int32_t getTensorVectorizedDim(char const* tensorName) const noexcept override;
        Dims getProfileShape(
            char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept override;
        int32_t getNbIOTensors() const noexcept override;
        char const* getIOTensorName(int32_t index) const noexcept override;
        HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept override;
        int32_t getNbAuxStreams() const noexcept override;

        int32_t getTensorBytesPerComponentV2(char const* tensorName, int32_t profileIndex) const noexcept override;
        int32_t getTensorComponentsPerElementV2(char const* tensorName, int32_t profileIndex) const noexcept override;
        TensorFormat getTensorFormatV2(char const* tensorName, int32_t profileIndex) const noexcept override;
        char const* getTensorFormatDescV2(char const* tensorName, int32_t profileIndex) const noexcept override;
        int32_t getTensorVectorizedDimV2(char const* tensorName, int32_t profileIndex) const noexcept override;

        ISerializationConfig* createSerializationConfig() noexcept override;
        IHostMemory* serializeWithConfig(ISerializationConfig& config) const noexcept override;

        size_t getDeviceMemorySizeForProfile(int32_t profileIndex) const noexcept override;
        IRefitter* createRefitter(ILogger& logger) noexcept override;

        bool setWeightStreamingBudget(int64_t gpuMemoryBudget) noexcept override;
        int64_t getWeightStreamingBudget() const noexcept override;
        int64_t getMinimumWeightStreamingBudget() const noexcept override;
        int64_t getStreamableWeightsSize() const noexcept override;

        bool isDebugTensor(char const* name) const noexcept override;

        // Added in TensorRT 10.1
        bool setWeightStreamingBudgetV2(int64_t gpuMemoryBudget) noexcept override;
        int64_t getWeightStreamingBudgetV2() const noexcept override;
        int64_t getWeightStreamingAutomaticBudget() const noexcept override;
        int64_t getWeightStreamingScratchMemorySize() const noexcept override;
        int64_t getDeviceMemorySizeV2() const noexcept override;
        int64_t getDeviceMemorySizeForProfileV2(int32_t profileIndex) const noexcept override;
        // Added in TensorRT 10.11
        TRT_NODISCARD int64_t const* getProfileTensorValuesV2(
            char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept override;
        TRT_NODISCARD IExecutionContext* createExecutionContextWithRuntimeConfig(
            IRuntimeConfig* runtimeConfig) noexcept override;
        TRT_NODISCARD IRuntimeConfig* createRuntimeConfig() noexcept override;

    private:
        std::shared_ptr<migraphx::program> mProgram;
        std::vector<std::string> mTensorNames;
        std::set<std::string> mOutputNames;
    };
} // ns:nvinfer1

#endif // NV_CUDA_ENGINE_IMPL_H
