#ifndef HELPER_HPP
#define HELPER_HPP

#include <migraphx/shape.hpp>
#include "migraphx/common_api/NvInfer.h"

namespace nvinfer1
{
namespace helper
{
    inline nvinfer1::DataType toDataType(const migraphx::shape::type_t& type)
    {
        switch(type)
        {
        case migraphx::shape::type_t::float_type: return nvinfer1::DataType::kFLOAT;
        case migraphx::shape::type_t::half_type: return nvinfer1::DataType::kHALF;
        case migraphx::shape::type_t::int8_type: return nvinfer1::DataType::kINT8;
        case migraphx::shape::type_t::int32_type: return nvinfer1::DataType::kINT32;
        case migraphx::shape::type_t::bool_type: return nvinfer1::DataType::kBOOL;
        case migraphx::shape::type_t::uint8_type: return nvinfer1::DataType::kUINT8;
        case migraphx::shape::type_t::fp8e4m3fnuz_type: return nvinfer1::DataType::kFP8;
        case migraphx::shape::type_t::int64_type: return nvinfer1::DataType::kINT64;
        default: MIGRAPHX_THROW("Type not supported");
        }
    }

    inline nvinfer1::Dims toDimensions(const migraphx::shape& shape)
    {
        nvinfer1::Dims dims;
        auto lens   = shape.lens();
        dims.nbDims = static_cast<int32_t>(lens.size());
        std::transform(
            lens.begin(), lens.end(), dims.d, [](auto l) { return static_cast<int64_t>(l); });
        return dims;
    }
}

}

#endif // HELPER_HPP