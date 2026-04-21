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

        case migraphx::shape::type_t::double_type:
        case migraphx::shape::type_t::uint16_type:
        case migraphx::shape::type_t::int16_type:
        case migraphx::shape::type_t::uint32_type:
        case migraphx::shape::type_t::uint64_type:
        case migraphx::shape::type_t::fp8e4m3fn_type:
        case migraphx::shape::type_t::fp8e5m2_type:
        case migraphx::shape::type_t::bf16_type:
        case migraphx::shape::type_t::fp8e5m2fnuz_type:
        case migraphx::shape::type_t::tuple_type:
        case migraphx::shape::type_t::fp4x2_type:
            MIGRAPHX_THROW("Type not supported");
        }
    }

    inline migraphx::shape::type_t fromDataType(const nvinfer1::DataType& type)
    {
        switch(type)
        {
        case nvinfer1::DataType::kFLOAT: return migraphx::shape::type_t::float_type;
        case nvinfer1::DataType::kHALF: return migraphx::shape::type_t::half_type;
        case nvinfer1::DataType::kINT8: return migraphx::shape::type_t::int8_type;
        case nvinfer1::DataType::kINT32: return migraphx::shape::type_t::int32_type;
        case nvinfer1::DataType::kBOOL: return migraphx::shape::type_t::bool_type;
        case nvinfer1::DataType::kUINT8: return migraphx::shape::type_t::uint8_type;
        case nvinfer1::DataType::kFP8: return migraphx::shape::type_t::fp8e4m3fnuz_type;
        case nvinfer1::DataType::kINT64: return migraphx::shape::type_t::int64_type;

        case nvinfer1::DataType::kBF16:
        case nvinfer1::DataType::kINT4:
        case nvinfer1::DataType::kFP4:
        case nvinfer1::DataType::kE8M0:
            MIGRAPHX_THROW("Type not supported");
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

    inline std::vector<int64_t> dimsToVec(const nvinfer1::Dims& dims)
    {
        std::vector<int64_t> ret;
        std::copy(dims.d, dims.d + dims.nbDims, std::back_inserter(ret));
        return ret;
    }
}

}

#endif // HELPER_HPP
