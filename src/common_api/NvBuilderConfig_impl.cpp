// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "migraphx/common_api/NvInferImpl.h"
#include "migraphx/common_api/NvInferRuntime.h"
#include "NvBuilderConfig_impl.hpp"

namespace nvinfer1
{
    NvBuilderConfig_impl::NvBuilderConfig_impl() noexcept
        : mBldrFlags{0}
    {
        mImpl = this;
    }

    NvBuilderConfig_impl::~NvBuilderConfig_impl()
    {
    }

    void NvBuilderConfig_impl::setAvgTimingIterations(int32_t avgTiming) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    int32_t NvBuilderConfig_impl::getAvgTimingIterations() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setEngineCapability(EngineCapability capability) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    EngineCapability NvBuilderConfig_impl::getEngineCapability() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return EngineCapability::kSTANDARD;
    }

    void NvBuilderConfig_impl::setInt8Calibrator(IInt8Calibrator* calibrator) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    IInt8Calibrator* NvBuilderConfig_impl::getInt8Calibrator() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    void NvBuilderConfig_impl::setFlags(BuilderFlags builderFlags) noexcept
    {
        mBldrFlags = builderFlags;
    }

    BuilderFlags NvBuilderConfig_impl::getFlags() const noexcept
    {
        return mBldrFlags;
    }

    void NvBuilderConfig_impl::clearFlag(BuilderFlag builderFlag) noexcept
    {
        mBldrFlags &= ~(1U << static_cast<uint32_t>(builderFlag));
    }

    void NvBuilderConfig_impl::setFlag(BuilderFlag builderFlag) noexcept
    {
        mBldrFlags |= (1U << static_cast<uint32_t>(builderFlag));
    }

    bool NvBuilderConfig_impl::getFlag(BuilderFlag builderFlag) const noexcept
    {
        return (mBldrFlags & (1U << static_cast<uint32_t>(builderFlag))) != 0;
    }

    void NvBuilderConfig_impl::setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    DeviceType NvBuilderConfig_impl::getDeviceType(ILayer const* layer) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return DeviceType::kGPU;
    }

    bool NvBuilderConfig_impl::isDeviceTypeSet(ILayer const* layer) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    void NvBuilderConfig_impl::resetDeviceType(ILayer const* layer) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    bool NvBuilderConfig_impl::canRunOnDLA(ILayer const* layer) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    void NvBuilderConfig_impl::setDLACore(int32_t dlaCore) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    int32_t NvBuilderConfig_impl::getDLACore() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setDefaultDeviceType(DeviceType deviceType) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    DeviceType NvBuilderConfig_impl::getDefaultDeviceType() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return DeviceType::kGPU;
    }

    void NvBuilderConfig_impl::reset() noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    void NvBuilderConfig_impl::setProfileStream(const hipStream_t stream) noexcept
    {
        pass_warning("TODO! implement me!", false);
    }

    hipStream_t NvBuilderConfig_impl::getProfileStream() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    int32_t NvBuilderConfig_impl::addOptimizationProfile(IOptimizationProfile const* profile) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    int32_t NvBuilderConfig_impl::getNbOptimizationProfiles() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    ProfilingVerbosity NvBuilderConfig_impl::getProfilingVerbosity() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return ProfilingVerbosity::kLAYER_NAMES_ONLY;
    }

    void NvBuilderConfig_impl::setAlgorithmSelector(IAlgorithmSelector* selector) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    IAlgorithmSelector* NvBuilderConfig_impl::getAlgorithmSelector() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    bool NvBuilderConfig_impl::setCalibrationProfile(IOptimizationProfile const* profile) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    IOptimizationProfile const* NvBuilderConfig_impl::getCalibrationProfile() noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    void NvBuilderConfig_impl::setQuantizationFlags(QuantizationFlags flags) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    QuantizationFlags NvBuilderConfig_impl::getQuantizationFlags() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::clearQuantizationFlag(QuantizationFlag flag) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    void NvBuilderConfig_impl::setQuantizationFlag(QuantizationFlag flag) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    bool NvBuilderConfig_impl::getQuantizationFlag(QuantizationFlag flag) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    bool NvBuilderConfig_impl::setTacticSources(TacticSources tacticSources) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    TacticSources NvBuilderConfig_impl::getTacticSources() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    nvinfer1::ITimingCache* NvBuilderConfig_impl::createTimingCache(void const* blob, std::size_t size) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    bool NvBuilderConfig_impl::setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    nvinfer1::ITimingCache const* NvBuilderConfig_impl::getTimingCache() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    void NvBuilderConfig_impl::setMemoryPoolLimit(MemoryPoolType pool, std::size_t poolSize) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    std::size_t NvBuilderConfig_impl::getMemoryPoolLimit(MemoryPoolType pool) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setPreviewFeature(PreviewFeature feature, bool enable) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    bool NvBuilderConfig_impl::getPreviewFeature(PreviewFeature feature) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    void NvBuilderConfig_impl::setBuilderOptimizationLevel(int32_t level) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    int32_t NvBuilderConfig_impl::getBuilderOptimizationLevel() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    HardwareCompatibilityLevel NvBuilderConfig_impl::getHardwareCompatibilityLevel() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return HardwareCompatibilityLevel::kNONE;
    }

    void NvBuilderConfig_impl::setPluginsToSerialize(char const* const* paths, int32_t nbPaths) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    char const* NvBuilderConfig_impl::getPluginToSerialize(int32_t index) const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    int32_t NvBuilderConfig_impl::getNbPluginsToSerialize() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setMaxAuxStreams(int32_t nbStreams) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    int32_t NvBuilderConfig_impl::getMaxAuxStreams() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    void NvBuilderConfig_impl::setProgressMonitor(IProgressMonitor* monitor) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    IProgressMonitor* NvBuilderConfig_impl::getProgressMonitor() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

    void NvBuilderConfig_impl::setRuntimePlatform(RuntimePlatform runtimePlatform) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    RuntimePlatform NvBuilderConfig_impl::getRuntimePlatform() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return RuntimePlatform::kSAME_AS_BUILD;
    }

    void NvBuilderConfig_impl::setMaxNbTactics(int32_t maxTactics) noexcept
    {
        pass_warning("TODO! implement me!", true);
    }

    int32_t NvBuilderConfig_impl::getMaxNbTactics() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    bool NvBuilderConfig_impl::setTilingOptimizationLevel(TilingOptimizationLevel level) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    TilingOptimizationLevel NvBuilderConfig_impl::getTilingOptimizationLevel() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return TilingOptimizationLevel::kNONE;
    }

    bool NvBuilderConfig_impl::setL2LimitForTiling(int64_t size) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    int64_t NvBuilderConfig_impl::getL2LimitForTiling() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return 0;
    }

    bool NvBuilderConfig_impl::setRemoteAutoTuningConfig(char const* config) noexcept
    {
        pass_warning("TODO! implement me!", true);
        return false;
    }

    char const* NvBuilderConfig_impl::getRemoteAutoTuningConfig() const noexcept
    {
        pass_warning("TODO! implement me!", true);
        return nullptr;
    }

} // ns:nvinfer1

