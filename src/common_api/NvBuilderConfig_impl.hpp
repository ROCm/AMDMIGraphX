#ifndef NV_BUILDER_CONFIG_IMPL_HPP
#define NV_BUILDER_CONFIG_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"

namespace nvinfer1
{
    class NvBuilderConfig_impl : public IBuilderConfig, public apiv::VBuilderConfig
    {
    public:
        NvBuilderConfig_impl() noexcept;
        ~NvBuilderConfig_impl() override;

        // public API
        void setAvgTimingIterations(int32_t avgTiming) noexcept override;
        int32_t getAvgTimingIterations() const noexcept override;
        void setEngineCapability(EngineCapability capability) noexcept override;
        EngineCapability getEngineCapability() const noexcept override;
        void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept override;
        IInt8Calibrator* getInt8Calibrator() const noexcept override;
        void setFlags(BuilderFlags builderFlags) noexcept override;
        BuilderFlags getFlags() const noexcept override;
        void clearFlag(BuilderFlag builderFlag) noexcept override;
        void setFlag(BuilderFlag builderFlag) noexcept override;
        bool getFlag(BuilderFlag builderFlag) const noexcept override;
        void setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept override;
        DeviceType getDeviceType(ILayer const* layer) const noexcept override;
        bool isDeviceTypeSet(ILayer const* layer) const noexcept override;
        void resetDeviceType(ILayer const* layer) noexcept override;
        bool canRunOnDLA(ILayer const* layer) const noexcept override;
        void setDLACore(int32_t dlaCore) noexcept override;
        int32_t getDLACore() const noexcept override;
        void setDefaultDeviceType(DeviceType deviceType) noexcept override;
        DeviceType getDefaultDeviceType() const noexcept override;
        void reset() noexcept override;
        void setProfileStream(const hipStream_t stream) noexcept override;
        hipStream_t getProfileStream() const noexcept override;
        int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept override;
        int32_t getNbOptimizationProfiles() const noexcept override;
        void setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept override;
        ProfilingVerbosity getProfilingVerbosity() const noexcept override;
        void setAlgorithmSelector(IAlgorithmSelector* selector) noexcept override;
        IAlgorithmSelector* getAlgorithmSelector() const noexcept override;
        bool setCalibrationProfile(IOptimizationProfile const* profile) noexcept override;
        IOptimizationProfile const* getCalibrationProfile() noexcept override;
        void setQuantizationFlags(QuantizationFlags flags) noexcept override;
        QuantizationFlags getQuantizationFlags() const noexcept override;
        void clearQuantizationFlag(QuantizationFlag flag) noexcept override;
        void setQuantizationFlag(QuantizationFlag flag) noexcept override;
        bool getQuantizationFlag(QuantizationFlag flag) const noexcept override;
        bool setTacticSources(TacticSources tacticSources) noexcept override;
        TacticSources getTacticSources() const noexcept override;
        nvinfer1::ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept override;
        bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept override;
        nvinfer1::ITimingCache const* getTimingCache() const noexcept override;
        void setMemoryPoolLimit(MemoryPoolType pool, std::size_t poolSize) noexcept override;
        std::size_t getMemoryPoolLimit(MemoryPoolType pool) const noexcept override;
        void setPreviewFeature(PreviewFeature feature, bool enable) noexcept override;
        bool getPreviewFeature(PreviewFeature feature) const noexcept override;
        void setBuilderOptimizationLevel(int32_t level) noexcept override;
        int32_t getBuilderOptimizationLevel() const noexcept override;
        void setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel) noexcept override;
        HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept override;
        void setPluginsToSerialize(char const* const* paths, int32_t nbPaths) noexcept override;
        char const* getPluginToSerialize(int32_t index) const noexcept override;
        int32_t getNbPluginsToSerialize() const noexcept override;
        void setMaxAuxStreams(int32_t nbStreams) noexcept override;
        int32_t getMaxAuxStreams() const noexcept override;
        void setProgressMonitor(IProgressMonitor* monitor) noexcept override;
        IProgressMonitor* getProgressMonitor() const noexcept override;
        void setRuntimePlatform(RuntimePlatform runtimePlatform) noexcept override;
        RuntimePlatform getRuntimePlatform() const noexcept override;
        void setMaxNbTactics(int32_t maxTactics) noexcept override;
        int32_t getMaxNbTactics() const noexcept override;
        bool setTilingOptimizationLevel(TilingOptimizationLevel level) noexcept override;
        TilingOptimizationLevel getTilingOptimizationLevel() const noexcept override;
        bool setL2LimitForTiling(int64_t size) noexcept override;
        int64_t getL2LimitForTiling() const noexcept override;
        bool setRemoteAutoTuningConfig(char const* config) noexcept override;
        char const* getRemoteAutoTuningConfig() const noexcept override;
    };

} // ns:nvinfer1

#endif // NV_BUILDER_CONFIG_IMPL_HPP
