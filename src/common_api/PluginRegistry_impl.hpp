#ifndef NV_PLUGIN_REGISTRY_IMPL_H
#define NV_PLUGIN_REGISTRY_IMPL_H

#include "migraphx/common_api/NvInferRuntimeCommon.h"

namespace nvinfer1
{

class PluginRegistry_impl : public IPluginRegistry
{
public:
    ~PluginRegistry_impl() override;

    // public API
    TRT_DEPRECATED bool registerCreator(IPluginCreator& creator, AsciiChar const* const pluginNamespace) noexcept override;
    TRT_DEPRECATED IPluginCreator* const* getPluginCreatorList(int32_t* const numCreators) const noexcept override;
    TRT_DEPRECATED IPluginCreator* getPluginCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace = "") noexcept override;
    void setErrorRecorder(IErrorRecorder* const recorder) noexcept override;
    IErrorRecorder* getErrorRecorder() const noexcept override;
    TRT_DEPRECATED bool deregisterCreator(IPluginCreator const& creator) noexcept override;
    bool isParentSearchEnabled() const override;
    void setParentSearchEnabled(bool const enabled) override;
    PluginLibraryHandle loadLibrary(AsciiChar const* pluginPath) noexcept override;
    void deregisterLibrary(PluginLibraryHandle handle) noexcept override;
    bool registerCreator(IPluginCreatorInterface& creator, AsciiChar const* const pluginNamespace) noexcept override;
    IPluginCreatorInterface* const* getAllCreators(int32_t* const numCreators) const noexcept override;
    IPluginCreatorInterface* getCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace = "") noexcept override;
    bool deregisterCreator(IPluginCreatorInterface const& creator) noexcept override;
    IPluginResource* acquirePluginResource(AsciiChar const* key, IPluginResource* resource) noexcept override;
    int32_t releasePluginResource(AsciiChar const* key) noexcept override;
    IPluginCreatorInterface* const* getAllCreatorsRecursive(int32_t* const numCreators) noexcept override;
};

}  // ns:nvinfer1

#endif // NV_PLUGIN_REGISTRY_H
