// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include <migraphx/onnx.hpp>

#include "NvOnnxParser_impl.hpp"

namespace nvonnxparser
{

NvOnnxParser_impl::NvOnnxParser_impl(void* network, void* logger, int version)
    :   mNetworkDefinition(static_cast<nvinfer1::NvNetworkDefinition_impl*>(network))
{
    pass_warning("TODO! implement me!", false);
}

NvOnnxParser_impl::~NvOnnxParser_impl()
{
    pass_warning("TODO! implement me!", false);
}

// public API
bool NvOnnxParser_impl::parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* /* model_path = nullptr */) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvOnnxParser_impl::parseFromFile(const char* onnxModelFile, int verbosity) noexcept 
{
    pass_warning("TODO! implement me!", false);
    try
    {
        mNetworkDefinition->setProgram(std::make_shared<migraphx::program>(migraphx::parse_onnx(onnxModelFile)));
    }
    catch(...)
    {
        return false;
    }

    return true;
}

TRT_DEPRECATED bool NvOnnxParser_impl::supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
                                            SubGraphCollection_t& sub_graph_collection, const char* model_path /* = nullptr */) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

TRT_DEPRECATED bool NvOnnxParser_impl::parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvOnnxParser_impl::supportsOperator(const char* op_name) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int NvOnnxParser_impl::getNbErrors() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return 0;
}
                                                                                               
IParserError const* NvOnnxParser_impl::getError(int index) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvOnnxParser_impl::clearErrors() noexcept 
{
    pass_warning("TODO! implement me!", true);
}

char const* const* NvOnnxParser_impl::getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

void NvOnnxParser_impl::setFlags(OnnxParserFlags onnxParserFlags) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

OnnxParserFlags NvOnnxParser_impl::getFlags() const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

void NvOnnxParser_impl::clearFlag(OnnxParserFlag onnxParserFlag) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

void NvOnnxParser_impl::setFlag(OnnxParserFlag onnxParserFlag) noexcept 
{
    pass_warning("TODO! implement me!", true);
}

bool NvOnnxParser_impl::getFlag(OnnxParserFlag onnxParserFlag) const noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

nvinfer1::ITensor const* NvOnnxParser_impl::getLayerOutputTensor(char const* name, int64_t i) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvOnnxParser_impl::supportsModelV2(void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath /* = nullptr */) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int64_t NvOnnxParser_impl::getNbSubgraphs() noexcept 
{
    pass_warning("TODO! implement me!", true);
    return 0;
}

bool NvOnnxParser_impl::isSubgraphSupported(int64_t const index) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

int64_t* NvOnnxParser_impl::getSubgraphNodes(int64_t const index, int64_t& subgraphLength) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

bool NvOnnxParser_impl::loadModelProto(void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath /* = nullptr */) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvOnnxParser_impl::loadInitializer(char const* name, void const* data, size_t size) noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

bool NvOnnxParser_impl::parseModelProto() noexcept 
{
    pass_warning("TODO! implement me!", true);
    return false;
}

}  // ns:nvinfer1
