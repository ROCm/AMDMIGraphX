#include "PluginRegistry_impl.hpp"

namespace nvinfer1
{

PluginRegistry_impl::~PluginRegistry_impl()
{
    // TODO! implement
}

TRT_DEPRECATED bool PluginRegistry_impl::registerCreator(IPluginCreator& creator, AsciiChar const* const pluginNamespace) noexcept 
{
    // TODO! implement
    return false;
}

TRT_DEPRECATED IPluginCreator* const* PluginRegistry_impl::getPluginCreatorList(int32_t* const numCreators) const noexcept 
{
    // TODO! implement
    return nullptr;
}

TRT_DEPRECATED IPluginCreator* PluginRegistry_impl::getPluginCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace /* = "" */) noexcept 
{
    // TODO! implement
    return nullptr;
}

void PluginRegistry_impl::setErrorRecorder(IErrorRecorder* const recorder) noexcept 
{
    // TODO! implement
}

IErrorRecorder* PluginRegistry_impl::getErrorRecorder() const noexcept 
{
    // TODO! implement
    return nullptr;
}

TRT_DEPRECATED bool PluginRegistry_impl::deregisterCreator(IPluginCreator const& creator) noexcept 
{
    // TODO! implement
    return false;
}

bool PluginRegistry_impl::isParentSearchEnabled() const 
{
    // TODO! implement
    return false;
}

void PluginRegistry_impl::setParentSearchEnabled(bool const enabled) 
{
    // TODO! implement
}

IPluginRegistry::PluginLibraryHandle PluginRegistry_impl::loadLibrary(AsciiChar const* pluginPath) noexcept 
{
    // TODO! implement
    return nullptr;
}

void PluginRegistry_impl::deregisterLibrary(PluginLibraryHandle handle) noexcept 
{
    // TODO! implement
}

bool PluginRegistry_impl::registerCreator(IPluginCreatorInterface& creator, AsciiChar const* const pluginNamespace) noexcept 
{
    // TODO! implement
    return false;
}

IPluginCreatorInterface* const* PluginRegistry_impl::getAllCreators(int32_t* const numCreators) const noexcept 
{
    // TODO! implement
    return nullptr;
}

IPluginCreatorInterface* PluginRegistry_impl::getCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion, AsciiChar const* const pluginNamespace /* = "" */) noexcept 
{
    // TODO! implement
    return nullptr;
}

bool PluginRegistry_impl::deregisterCreator(IPluginCreatorInterface const& creator) noexcept 
{
    // TODO! implement
    return false;
}

IPluginResource* PluginRegistry_impl::acquirePluginResource(AsciiChar const* key, IPluginResource* resource) noexcept 
{
    // TODO! implement
    return nullptr;
}

int32_t PluginRegistry_impl::releasePluginResource(AsciiChar const* key) noexcept 
{
    // TODO! implement
    return 0;
}

IPluginCreatorInterface* const* PluginRegistry_impl::getAllCreatorsRecursive(int32_t* const numCreators) noexcept 
{
    // TODO! implement
    return nullptr;
}

}  // ns:nvinfer1
