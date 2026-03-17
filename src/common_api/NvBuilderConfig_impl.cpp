#include "migraphx/common_api/NvInferImpl.h"
#include "migraphx/common_api/NvInferRuntime.h"
#include "NvBuilderConfig_impl.hpp"

namespace nvinfer1
{
    NvBuilderConfig_impl::NvBuilderConfig_impl() noexcept
    {
        // TODO! implement
        mImpl = this;
    }

    NvBuilderConfig_impl::~NvBuilderConfig_impl()
    {
        // TODO! implement
    }

    void NvBuilderConfig_impl::setAvgTimingIterations(int32_t avgTiming) noexcept
    {
        // TODO! implement
    }

    int32_t NvBuilderConfig_impl::getAvgTimingIterations() const noexcept
    {
        // TODO! implement
        return 0;
    }

    void NvBuilderConfig_impl::setEngineCapability(EngineCapability capability) noexcept
    {
        // TODO! implement
    }

    EngineCapability NvBuilderConfig_impl::getEngineCapability() const noexcept
    {
        // TODO! implement
        return EngineCapability::kSTANDARD;
    }

    void NvBuilderConfig_impl::setInt8Calibrator(IInt8Calibrator* calibrator) noexcept
    {
        // TODO! implement
    }

    IInt8Calibrator* NvBuilderConfig_impl::getInt8Calibrator() const noexcept
    {
        // TODO! implement
        return nullptr;
    }

    void NvBuilderConfig_impl::setFlags(BuilderFlags builderFlags) noexcept
    {
        // TODO! implement
    }

    BuilderFlags NvBuilderConfig_impl::getFlags() const noexcept
    {
        // TODO! implement
    }

    void NvBuilderConfig_impl::clearFlag(BuilderFlag builderFlag) noexcept
    {
        // TODO! implement
    }

    void NvBuilderConfig_impl::setFlag(BuilderFlag builderFlag) noexcept
    {
        // TODO! implement
    }

    bool NvBuilderConfig_impl::getFlag(BuilderFlag builderFlag) const noexcept
    {
        // TODO! implement
        return false;
    }

    void NvBuilderConfig_impl::setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept
    {
        // TODO! implement
    }

    DeviceType NvBuilderConfig_impl::getDeviceType(ILayer const* layer) const noexcept
    {
        // TODO! implement
    }

    bool NvBuilderConfig_impl::isDeviceTypeSet(ILayer const* layer) const noexcept
    {
		// TODO! implement
        return false;
	}

    void NvBuilderConfig_impl::resetDeviceType(ILayer const* layer) noexcept
    {
		// TODO! implement
	}

    bool NvBuilderConfig_impl::canRunOnDLA(ILayer const* layer) const noexcept
    {
		// TODO! implement
        return false;
	}

    void NvBuilderConfig_impl::setDLACore(int32_t dlaCore) noexcept
    {
		// TODO! implement
	}

    int32_t NvBuilderConfig_impl::getDLACore() const noexcept
    {
		// TODO! implement
        return 0;
	}

    void NvBuilderConfig_impl::setDefaultDeviceType(DeviceType deviceType) noexcept
    {
		// TODO! implement
	}

    DeviceType NvBuilderConfig_impl::getDefaultDeviceType() const noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::reset() noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::setProfileStream(const hipStream_t stream) noexcept
    {
		// TODO! implement
	}

    hipStream_t NvBuilderConfig_impl::getProfileStream() const noexcept
    {
		// TODO! implement
	}

    int32_t NvBuilderConfig_impl::addOptimizationProfile(IOptimizationProfile const* profile) noexcept
    {
		// TODO! implement
        return 0;
	}

    int32_t NvBuilderConfig_impl::getNbOptimizationProfiles() const noexcept
    {
		// TODO! implement
        return 0;
	}

    void NvBuilderConfig_impl::setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept
    {
		// TODO! implement
	}

    ProfilingVerbosity NvBuilderConfig_impl::getProfilingVerbosity() const noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::setAlgorithmSelector(IAlgorithmSelector* selector) noexcept
    {
		// TODO! implement
	}

    IAlgorithmSelector* NvBuilderConfig_impl::getAlgorithmSelector() const noexcept
    {
		// TODO! implement
        return nullptr;
	}

    bool NvBuilderConfig_impl::setCalibrationProfile(IOptimizationProfile const* profile) noexcept
    {
		// TODO! implement
        return false;
	}

    IOptimizationProfile const* NvBuilderConfig_impl::getCalibrationProfile() noexcept
    {
		// TODO! implement
        return nullptr;
	}

    void NvBuilderConfig_impl::setQuantizationFlags(QuantizationFlags flags) noexcept
    {
		// TODO! implement
	}

    QuantizationFlags NvBuilderConfig_impl::getQuantizationFlags() const noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::clearQuantizationFlag(QuantizationFlag flag) noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::setQuantizationFlag(QuantizationFlag flag) noexcept
    {
		// TODO! implement
	}

    bool NvBuilderConfig_impl::getQuantizationFlag(QuantizationFlag flag) const noexcept
    {
		// TODO! implement
        return false;
	}

    bool NvBuilderConfig_impl::setTacticSources(TacticSources tacticSources) noexcept
    {
		// TODO! implement
        return false;
	}

    TacticSources NvBuilderConfig_impl::getTacticSources() const noexcept
    {
		// TODO! implement
	}

    nvinfer1::ITimingCache* NvBuilderConfig_impl::createTimingCache(void const* blob, std::size_t size) const noexcept
    {
		// TODO! implement
        return nullptr;
	}

    bool NvBuilderConfig_impl::setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept
    {
		// TODO! implement
        return false;
	}

    nvinfer1::ITimingCache const* NvBuilderConfig_impl::getTimingCache() const noexcept
    {
		// TODO! implement
        return nullptr;
	}

    void NvBuilderConfig_impl::setMemoryPoolLimit(MemoryPoolType pool, std::size_t poolSize) noexcept
    {
		// TODO! implement
	}

    std::size_t NvBuilderConfig_impl::getMemoryPoolLimit(MemoryPoolType pool) const noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::setPreviewFeature(PreviewFeature feature, bool enable) noexcept
    {
		// TODO! implement
	}

    bool NvBuilderConfig_impl::getPreviewFeature(PreviewFeature feature) const noexcept
    {
		// TODO! implement
        return false;
	}

    void NvBuilderConfig_impl::setBuilderOptimizationLevel(int32_t level) noexcept
    {
		// TODO! implement
	}

    int32_t NvBuilderConfig_impl::getBuilderOptimizationLevel() const noexcept
    {
		// TODO! implement
        return 0;
	}

    void NvBuilderConfig_impl::setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel) noexcept
    {
		// TODO! implement
	}

    HardwareCompatibilityLevel NvBuilderConfig_impl::getHardwareCompatibilityLevel() const noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::setPluginsToSerialize(char const* const* paths, int32_t nbPaths) noexcept
    {
		// TODO! implement
	}

    char const* NvBuilderConfig_impl::getPluginToSerialize(int32_t index) const noexcept
    {
		// TODO! implement
        return nullptr;
	}

    int32_t NvBuilderConfig_impl::getNbPluginsToSerialize() const noexcept
    {
		// TODO! implement
        return 0;
	}

    void NvBuilderConfig_impl::setMaxAuxStreams(int32_t nbStreams) noexcept
    {
		// TODO! implement
	}

    int32_t NvBuilderConfig_impl::getMaxAuxStreams() const noexcept
    {
		// TODO! implement
        return 0;
	}

    void NvBuilderConfig_impl::setProgressMonitor(IProgressMonitor* monitor) noexcept
    {
		// TODO! implement
	}

    IProgressMonitor* NvBuilderConfig_impl::getProgressMonitor() const noexcept
    {
		// TODO! implement
        return nullptr;
	}

    void NvBuilderConfig_impl::setRuntimePlatform(RuntimePlatform runtimePlatform) noexcept
    {
		// TODO! implement
	}

    RuntimePlatform NvBuilderConfig_impl::getRuntimePlatform() const noexcept
    {
		// TODO! implement
	}

    void NvBuilderConfig_impl::setMaxNbTactics(int32_t maxTactics) noexcept
    {
		// TODO! implement
	}

    int32_t NvBuilderConfig_impl::getMaxNbTactics() const noexcept
    {
		// TODO! implement
        return 0;
	}

    bool NvBuilderConfig_impl::setTilingOptimizationLevel(TilingOptimizationLevel level) noexcept
    {
		// TODO! implement
        return false;
	}

    TilingOptimizationLevel NvBuilderConfig_impl::getTilingOptimizationLevel() const noexcept
    {
		// TODO! implement
	}

    bool NvBuilderConfig_impl::setL2LimitForTiling(int64_t size) noexcept
    {
		// TODO! implement
        return false;
	}

    int64_t NvBuilderConfig_impl::getL2LimitForTiling() const noexcept
    {
		// TODO! implement
        return 0;
	}

    bool NvBuilderConfig_impl::setRemoteAutoTuningConfig(char const* config) noexcept
    {
		// TODO! implement
        return false;
	}

    char const* NvBuilderConfig_impl::getRemoteAutoTuningConfig() const noexcept
    {
		// TODO! implement
        return nullptr;
	}

} // ns:nvinfer1

