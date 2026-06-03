// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/register_target.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/iterator_for.hpp>

#include "migraphx/common_api/NvInferImpl.h"
#include "migraphx/common_api/NvInferRuntime.h"

#include "NvBuilder_impl.hpp"
#include "NvHostMemory_impl.hpp"
#include "NetworkDefinition_impl.hpp"
#include "NvBuilderConfig_impl.hpp"

namespace nvinfer1
{

NvBuilder_impl::NvBuilder_impl(void* logger, int32_t version) noexcept
    :  mPluginRegistry(nullptr), mNetworkDefinition(nullptr)
{
    pass_warning("TODO! implement me!", false);
    mImpl = this;
}

NvBuilder_impl::~NvBuilder_impl()
{
    pass_warning("TODO! implement me!", false);
}

bool NvBuilder_impl::platformHasFastFp16() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return true;
}

bool NvBuilder_impl::platformHasFastInt8() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return true;
}

int32_t NvBuilder_impl::getMaxDLABatchSize() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

int32_t NvBuilder_impl::getNbDLACores() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void NvBuilder_impl::setGpuAllocator(IGpuAllocator* allocator) noexcept
{
    pass_warning("TODO! implement me!", true);
}

nvinfer1::IBuilderConfig* NvBuilder_impl::createBuilderConfig() noexcept
{
    mBuilderConfig = std::make_unique<NvBuilderConfig_impl>();
    return mBuilderConfig.release();
}

nvinfer1::INetworkDefinition* NvBuilder_impl::createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept
{
    mNetworkDefinition = std::make_unique<NvNetworkDefinition_impl>(flags, *this);
    return mNetworkDefinition.release();
}

nvinfer1::IOptimizationProfile* NvBuilder_impl::createOptimizationProfile() noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvBuilder_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{    
}

IErrorRecorder* NvBuilder_impl::getErrorRecorder() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvBuilder_impl::reset() noexcept
{
    pass_warning("TODO! implement me!", true);
}

bool NvBuilder_impl::platformHasTf32() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

nvinfer1::IHostMemory* NvBuilder_impl::buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept
{
    auto& nw_impl = static_cast<NvNetworkDefinition_impl&>(network);
    
    nw_impl.build();
    
    migraphx::program prog = *nw_impl.getProgram();
    if(std::getenv("COMMONAPI_DEBUG_PROGRAM") != nullptr)
    {
        std::cout << "==== program before compile ====\n" << prog << std::endl;
    }
    try
    {
        prog.compile(migraphx::make_target("gpu"));
    }
    catch(migraphx::exception& /*e*/)
    {
        // TODO write to error recorder/logger
        return nullptr;
    }

    // replace_allocate names the generated output parameters "<module>:#output_N".
    // Expose them instead under the names of the tensors that were marked as
    // network outputs (TensorRT-style binding names), e.g. "output".
    {
        const auto output_names = nw_impl.getOutputNames();
        const std::string prefix = "#output_";
        auto* mm                 = prog.get_main_module();
        for(auto ins : migraphx::iterator_for(*mm))
        {
            if(ins->name() != "@param")
                continue;
            const std::string param_name = migraphx::any_cast<migraphx::builtin::param>(ins->get_operator()).parameter;
            auto loc = param_name.find(prefix);
            if(loc == std::string::npos)
                continue;
            try
            {
                const std::size_t index = std::stoul(param_name.substr(loc + prefix.size()));
                if(index < output_names.size() and not output_names[index].empty())
                    mm->rename_parameter(ins, output_names[index]);
            }
            catch(...)
            {
                // leave the parameter name unchanged
            }
        }
    }

    // Parameter ordering is significant for loop body submodules (run_loop binds
    // body parameters positionally via get_parameter_names()). MIGraphX does not
    // serialize the parameter "order" field, so on reload the order is rebuilt from
    // the physical instruction order. Compilation can leave parameters physically
    // out of order (e.g. the loop condition parameter, consumed only by @return,
    // ends up last). Reorder each module's parameter instructions to match their
    // logical order so the order survives the save/load round-trip.
    for(auto* mod : prog.get_modules())
    {
        const auto names = mod->get_parameter_names();
        for(auto it = names.rbegin(); it != names.rend(); ++it)
        {
            auto param = mod->get_parameter(*it);
            if(param != mod->end())
                mod->move_instruction(param, mod->begin());
        }
    }

    mSerializedNetworks.push_back(migraphx::save_buffer(prog));
    auto&& current_network = mSerializedNetworks.back();

    return new NvHostMemory_impl{reinterpret_cast<void*>(current_network.data()),
                            current_network.size(),
                            DataType::kUINT8};
}

bool NvBuilder_impl::isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

ILogger* NvBuilder_impl::getLogger() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvBuilder_impl::setMaxThreads(int32_t maxThreads) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int32_t NvBuilder_impl::getMaxThreads() const noexcept
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

IPluginRegistry& NvBuilder_impl::getPluginRegistry() noexcept
{
    pass_warning("TODO! implement me!", true);
    return *reinterpret_cast<IPluginRegistry*>(mPluginRegistry);
}

ICudaEngine* NvBuilder_impl::buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvBuilder_impl::buildSerializedNetworkToStream(
    INetworkDefinition& network, IBuilderConfig& config, IStreamWriter& writer) noexcept
{
    pass_warning("TODO! implement me!", true);
    return false;
}

nvinfer1::IHostMemory* NvBuilder_impl::buildSerializedNetworkWithKernelText(
    INetworkDefinition& network, IBuilderConfig& config, IHostMemory*& kernelText) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

}  // ns:nvinfer1
