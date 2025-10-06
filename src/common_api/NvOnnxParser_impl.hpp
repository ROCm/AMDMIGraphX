#ifndef NV_ONNX_PARSER_IMPL_H
#define NV_ONNX_PARSER_IMPL_H

#include "migraphx/common_api/NvOnnxParser.h"

namespace nvonnxparser
{
    class NvOnnxParser_impl : public IParser
    {
    public:
        NvOnnxParser_impl(void* network, void* logger, int version);
        ~NvOnnxParser_impl() override;

        // public API
        bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path = nullptr) noexcept override;
        bool parseFromFile(const char* onnxModelFile, int verbosity) noexcept override;
        TRT_DEPRECATED bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
                                                    SubGraphCollection_t& sub_graph_collection, const char* model_path = nullptr) noexcept override;
        TRT_DEPRECATED bool parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size) noexcept override;
        bool supportsOperator(const char* op_name) const noexcept override;
        int getNbErrors() const noexcept override;                                                                                               
        IParserError const* getError(int index) const noexcept override;
        void clearErrors() noexcept override;
        char const* const* getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept override;
        void setFlags(OnnxParserFlags onnxParserFlags) noexcept override;
        OnnxParserFlags getFlags() const noexcept override;
        void clearFlag(OnnxParserFlag onnxParserFlag) noexcept override;
        void setFlag(OnnxParserFlag onnxParserFlag) noexcept override;
        bool getFlag(OnnxParserFlag onnxParserFlag) const noexcept override;
        nvinfer1::ITensor const* getLayerOutputTensor(char const* name, int64_t i) noexcept override;
        bool supportsModelV2(void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath = nullptr) noexcept override;
        int64_t getNbSubgraphs() noexcept override;
        bool isSubgraphSupported(int64_t const index) noexcept override;
        int64_t* getSubgraphNodes(int64_t const index, int64_t& subgraphLength) noexcept override;
        bool loadModelProto(void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath = nullptr) noexcept override;
        bool loadInitializer(char const* name, void const* data, size_t size) noexcept override;
        bool parseModelProto() noexcept override;
    };

}   // ns:nvonnxparser

#endif // NV_ONNX_PARSER_IMPL_H
