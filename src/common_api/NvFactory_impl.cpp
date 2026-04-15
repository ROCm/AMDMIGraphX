// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "migraphx/common_api/NvInferRuntime.h"

#include "NvBuilder_impl.hpp"
#include "NvOnnxParser_impl.hpp"
#include "NvRuntime_impl.hpp"

extern "C" TENSORRTAPI void* createInferBuilder_INTERNAL(void* logger, int32_t version) noexcept
{
    return new nvinfer1::NvBuilder_impl(logger, version);
}

extern "C" TENSORRTAPI void* createInferRefitter_INTERNAL(void* engine, void* logger, int32_t version) noexcept
{
    pass_warning("TODO! implement me!", true);
    return nullptr;
}

extern "C" TENSORRTAPI void* createNvOnnxParser_INTERNAL(void* network, void* logger, int version) noexcept
{
    try
    {
        return new nvonnxparser::NvOnnxParser_impl(network, logger, version);
    }
    catch(...)
    {
        return nullptr;
    }
}

extern "C" TENSORRTAPI void* createInferRuntime_INTERNAL(void* logger, int32_t version) noexcept
{
    return new nvinfer1::NvRuntime_impl(logger, version);
}

