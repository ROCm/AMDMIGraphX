#include "NvOnnxParser_impl.hpp"
#include "migraphx/common_api/NvInferRuntime.h"

#include "NvBuilder_impl.hpp"

extern "C" TENSORRTAPI void* createInferBuilder_INTERNAL(void* logger, int32_t version) noexcept
{
    return new nvinfer1::NvBuilder_impl(logger, version);
}

extern "C" TENSORRTAPI void* createInferRefitter_INTERNAL(void* engine, void* logger, int32_t version) noexcept
{
    return nullptr;
}

extern "C" TENSORRTAPI void* createNvOnnxParser_INTERNAL(void* network, void* logger, int version) noexcept
{
    static nvonnxparser::NvOnnxParser_impl parser(network, logger, version);
    return &parser;
}

extern "C" TENSORRTAPI void* createInferRuntime_INTERNAL(void* logger, int32_t version) noexcept
{
    return nullptr;
}

