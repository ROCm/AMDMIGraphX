// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "PluginRegistry_impl.hpp"

namespace nvinfer1
{

PluginRegistry_impl::~PluginRegistry_impl()
{
    pass_warning("TODO! implement me!", false);
}

TRT_DEPRECATED bool PluginRegistry_impl::registerCreator(IPluginCreator& creator, AsciiChar const* const pluginNamespace) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

TRT_DEPRECATED IPluginCreator* const* PluginRegistry_impl::getPluginCreatorList(int32_t* const numCreators) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

TRT_DEPRECATED IPluginCreator* PluginRegistry_impl::getPluginCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace /* = "" */) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void PluginRegistry_impl::setErrorRecorder(IErrorRecorder* const recorder) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

IErrorRecorder* PluginRegistry_impl::getErrorRecorder() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

TRT_DEPRECATED bool PluginRegistry_impl::deregisterCreator(IPluginCreator const& creator) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool PluginRegistry_impl::isParentSearchEnabled() const 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

void PluginRegistry_impl::setParentSearchEnabled(bool const enabled) 
{
    pass_warning("TODO! implement me!", true);
}

IPluginRegistry::PluginLibraryHandle PluginRegistry_impl::loadLibrary(AsciiChar const* pluginPath) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void PluginRegistry_impl::deregisterLibrary(PluginLibraryHandle handle) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

bool PluginRegistry_impl::registerCreator(IPluginCreatorInterface& creator, AsciiChar const* const pluginNamespace) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

IPluginCreatorInterface* const* PluginRegistry_impl::getAllCreators(int32_t* const numCreators) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

IPluginCreatorInterface* PluginRegistry_impl::getCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace /* = "" */) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool PluginRegistry_impl::deregisterCreator(IPluginCreatorInterface const& creator) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

IPluginResource* PluginRegistry_impl::acquirePluginResource(AsciiChar const* key, IPluginResource* resource) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

int32_t PluginRegistry_impl::releasePluginResource(AsciiChar const* key) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

IPluginCreatorInterface* const* PluginRegistry_impl::getAllCreatorsRecursive(int32_t* const numCreators) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

}  // ns:nvinfer1
