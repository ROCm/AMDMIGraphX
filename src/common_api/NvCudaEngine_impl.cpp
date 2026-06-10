// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/ranges.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>

#include "migraphx/common_api/NvInfer.h"
#include "NvCudaEngine_impl.hpp"
#include "NvExecutionContext_impl.hpp"
#include "Helper.hpp"

namespace nvinfer1
{

namespace
{
    // Try to read the parameter name of a @param instruction; return "" otherwise.
    std::string param_name_of(migraphx::instruction_ref ins)
    {
        if(ins->name() != "@param")
            return {};
        return migraphx::any_cast<migraphx::builtin::param>(ins->get_operator()).parameter;
    }

    // Determine which parameters of the compiled program are network outputs.
    //
    // The substring "output" heuristic alone is not reliable: callers may name an
    // output anything (e.g. an ITopKLayer produces "values"/"indices"). After
    // compilation MIGraphX materializes each output into a parameter buffer that is
    // the write destination - i.e. the LAST argument of an instruction feeding the
    // module's @return (or a parameter returned directly). Input parameters are only
    // ever read and never appear in that trailing output-buffer position.
    std::set<std::string> compute_output_names(const migraphx::program& program)
    {
        std::set<std::string> outputs;
        const auto* mm = program.get_main_module();
        if(mm == nullptr or mm->begin() == mm->end())
            return outputs;

        auto ret = std::prev(mm->end());
        if(ret->name() != "@return")
            return outputs;

        for(auto arg : ret->inputs())
        {
            // A parameter returned directly (identity passthrough).
            auto direct = param_name_of(arg);
            if(not direct.empty())
            {
                outputs.insert(direct);
                continue;
            }
            // Otherwise the output buffer is the producing instruction's last argument.
            const auto& sub = arg->inputs();
            if(not sub.empty())
            {
                auto dst = param_name_of(sub.back());
                if(not dst.empty())
                    outputs.insert(dst);
            }
        }
        return outputs;
    }
}  // namespace

NvCudaEngine_impl::NvCudaEngine_impl(void* logger, int32_t version) noexcept
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

NvCudaEngine_impl::NvCudaEngine_impl(const std::shared_ptr<migraphx::program>& program) noexcept
    : mProgram(program), mTensorNames{program->get_parameter_names()}, mOutputNames{compute_output_names(*program)}
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

NvCudaEngine_impl::~NvCudaEngine_impl()
{
    pass_warning("TODO! implement me!", false);
}

ICudaEngine* NvCudaEngine_impl::getPImpl() noexcept
{
    pass_warning("TODO! implement me!", true);
    return this;
}

int32_t NvCudaEngine_impl::getNbLayers() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

IHostMemory* NvCudaEngine_impl::serialize() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

IExecutionContext* NvCudaEngine_impl::createExecutionContext(ExecutionContextAllocationStrategy strategy) noexcept
{
    return new NvExecutionContext_impl{mProgram};
}

IExecutionContext* NvCudaEngine_impl::createExecutionContextWithoutDeviceMemory() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

size_t NvCudaEngine_impl::getDeviceMemorySize() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvCudaEngine_impl::isRefittable() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

char const* NvCudaEngine_impl::getName() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

int32_t NvCudaEngine_impl::getNbOptimizationProfiles() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t const* NvCudaEngine_impl::getProfileTensorValues(
    char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

EngineCapability NvCudaEngine_impl::getEngineCapability() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return EngineCapability::kSTANDARD;
}

void NvCudaEngine_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
    pass_warning("TODO! implement me!", true);
}

IErrorRecorder* NvCudaEngine_impl::getErrorRecorder() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvCudaEngine_impl::hasImplicitBatchDimension() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

TacticSources NvCudaEngine_impl::getTacticSources() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

ProfilingVerbosity NvCudaEngine_impl::getProfilingVerbosity() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return ProfilingVerbosity::kLAYER_NAMES_ONLY;
}

IEngineInspector* NvCudaEngine_impl::createEngineInspector() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

Dims NvCudaEngine_impl::getTensorShape(char const* tensorName) const noexcept
{
    return helper::toDimensions(mProgram->get_parameter_shapes().at(tensorName));
}

DataType NvCudaEngine_impl::getTensorDataType(char const* tensorName) const noexcept
{
    return helper::toDataType(mProgram->get_parameter_shapes().at(tensorName).type());
}

TensorLocation NvCudaEngine_impl::getTensorLocation(char const* tensorName) const noexcept
{   
    pass_warning("TODO! implement me!", true);
    return TensorLocation::kDEVICE;
}

bool NvCudaEngine_impl::isShapeInferenceIO(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

TensorIOMode NvCudaEngine_impl::getTensorIOMode(char const* tensorName) const noexcept
{
    const std::string name{tensorName};
    // A parameter is an output if it was detected as a network output of the
    // compiled program, or (as a fallback) its name carries the default
    // "#output_N" / "output" binding token.
    if(mOutputNames.count(name) != 0 or migraphx::contains(name, "output"))
        return TensorIOMode::kOUTPUT;
    return TensorIOMode::kINPUT;
}

int32_t NvCudaEngine_impl::getTensorBytesPerComponent(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t NvCudaEngine_impl::getTensorComponentsPerElement(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

TensorFormat NvCudaEngine_impl::getTensorFormat(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return TensorFormat::kLINEAR;
}

char const* NvCudaEngine_impl::getTensorFormatDesc(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

int32_t NvCudaEngine_impl::getTensorVectorizedDim(char const* tensorName) const noexcept
{
    pass_warning("TODO! implement me!", false);
    return -1;
}

Dims NvCudaEngine_impl::getProfileShape(
    char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return Dims{};
}

int32_t NvCudaEngine_impl::getNbIOTensors() const noexcept
{
    return mTensorNames.size();
}

char const* NvCudaEngine_impl::getIOTensorName(int32_t index) const noexcept
{
    return mTensorNames.at(index).c_str();
}

HardwareCompatibilityLevel NvCudaEngine_impl::getHardwareCompatibilityLevel() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return HardwareCompatibilityLevel::kNONE;
}

int32_t NvCudaEngine_impl::getNbAuxStreams() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t NvCudaEngine_impl::getTensorBytesPerComponentV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t NvCudaEngine_impl::getTensorComponentsPerElementV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

TensorFormat NvCudaEngine_impl::getTensorFormatV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return TensorFormat::kLINEAR;
}

char const* NvCudaEngine_impl::getTensorFormatDescV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

int32_t NvCudaEngine_impl::getTensorVectorizedDimV2(char const* tensorName, int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return -1;
}

ISerializationConfig* NvCudaEngine_impl::createSerializationConfig() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

IHostMemory* NvCudaEngine_impl::serializeWithConfig(ISerializationConfig& config) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

size_t NvCudaEngine_impl::getDeviceMemorySizeForProfile(int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

IRefitter* NvCudaEngine_impl::createRefitter(ILogger& logger) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvCudaEngine_impl::setWeightStreamingBudget(int64_t gpuMemoryBudget) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int64_t NvCudaEngine_impl::getWeightStreamingBudget() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int64_t NvCudaEngine_impl::getMinimumWeightStreamingBudget() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int64_t NvCudaEngine_impl::getStreamableWeightsSize() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvCudaEngine_impl::isDebugTensor(char const* name) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

// Added in TensorRT 10.1
bool NvCudaEngine_impl::setWeightStreamingBudgetV2(int64_t gpuMemoryBudget) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int64_t NvCudaEngine_impl::getWeightStreamingBudgetV2() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int64_t NvCudaEngine_impl::getWeightStreamingAutomaticBudget() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int64_t NvCudaEngine_impl::getWeightStreamingScratchMemorySize() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int64_t NvCudaEngine_impl::getDeviceMemorySizeV2() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int64_t NvCudaEngine_impl::getDeviceMemorySizeForProfileV2(int32_t profileIndex) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

// Added in TensorRT 10.11
TRT_NODISCARD int64_t const* NvCudaEngine_impl::getProfileTensorValuesV2(
    char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

TRT_NODISCARD IExecutionContext* NvCudaEngine_impl::createExecutionContextWithRuntimeConfig(
    IRuntimeConfig* runtimeConfig) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

TRT_NODISCARD IRuntimeConfig* NvCudaEngine_impl::createRuntimeConfig() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr; 
}

} // ns:nvinfer1
