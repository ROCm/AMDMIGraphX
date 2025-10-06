#include "NvOnnxParser_impl.hpp"

namespace nvonnxparser
{

NvOnnxParser_impl::NvOnnxParser_impl(void* network, void* logger, int version)
{
    // TODO! implement
}

NvOnnxParser_impl::~NvOnnxParser_impl()
{
    // TODO! implement
}

// public API
bool NvOnnxParser_impl::parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* /* model_path = nullptr */) noexcept 
{
    // TODO! implement
    return false;
}

bool NvOnnxParser_impl::parseFromFile(const char* onnxModelFile, int verbosity) noexcept 
{
    // TODO! implement
    return false;
}

TRT_DEPRECATED bool NvOnnxParser_impl::supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
                                            SubGraphCollection_t& sub_graph_collection, const char* model_path /* = nullptr */) noexcept 
{
    // TODO! implement
    return false;
}

TRT_DEPRECATED bool NvOnnxParser_impl::parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size) noexcept 
{
    // TODO! implement
    return false;
}

bool NvOnnxParser_impl::supportsOperator(const char* op_name) const noexcept 
{
    // TODO! implement
    return false;
}

int NvOnnxParser_impl::getNbErrors() const noexcept 
{
    // TODO! implement
    return 0;
}
                                                                                               
IParserError const* NvOnnxParser_impl::getError(int index) const noexcept 
{
    // TODO! implement
    return nullptr;
}

void NvOnnxParser_impl::clearErrors() noexcept 
{
    // TODO! implement
}

char const* const* NvOnnxParser_impl::getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept 
{
    // TODO! implement
    return nullptr;
}

void NvOnnxParser_impl::setFlags(OnnxParserFlags onnxParserFlags) noexcept 
{
    // TODO! implement
}

OnnxParserFlags NvOnnxParser_impl::getFlags() const noexcept 
{
    // TODO! implement
}

void NvOnnxParser_impl::clearFlag(OnnxParserFlag onnxParserFlag) noexcept 
{
    // TODO! implement
}

void NvOnnxParser_impl::setFlag(OnnxParserFlag onnxParserFlag) noexcept 
{
    // TODO! implement
}

bool NvOnnxParser_impl::getFlag(OnnxParserFlag onnxParserFlag) const noexcept 
{
    // TODO! implement
    return false;
}

nvinfer1::ITensor const* NvOnnxParser_impl::getLayerOutputTensor(char const* name, int64_t i) noexcept 
{
    // TODO! implement
    return nullptr;
}

bool NvOnnxParser_impl::supportsModelV2(void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath /* = nullptr */) noexcept 
{
    // TODO! implement
    return false;
}

int64_t NvOnnxParser_impl::getNbSubgraphs() noexcept 
{
    // TODO! implement
    return 0;
}

bool NvOnnxParser_impl::isSubgraphSupported(int64_t const index) noexcept 
{
    // TODO! implement
    return false;
}

int64_t* NvOnnxParser_impl::getSubgraphNodes(int64_t const index, int64_t& subgraphLength) noexcept 
{
    // TODO! implement
    return nullptr;
}

bool NvOnnxParser_impl::loadModelProto(void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath /* = nullptr */) noexcept 
{
    // TODO! implement
    return false;
}

bool NvOnnxParser_impl::loadInitializer(char const* name, void const* data, size_t size) noexcept 
{
    // TODO! implement
    return false;
}

bool NvOnnxParser_impl::parseModelProto() noexcept 
{
    // TODO! implement
    return false;
}

}  // ns:nvinfer1
