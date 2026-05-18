#ifndef TENSOR_IMPL_HPP
#define TENSOR_IMPL_HPP

#include "migraphx/common_api/NvInfer.h"

#include <migraphx/instruction.hpp>

namespace nvinfer1
{
    class Tensor_impl : public ITensor, public apiv::VTensor
    {
    public:
        Tensor_impl() noexcept;
        Tensor_impl(migraphx::instruction_ref ins) noexcept;
        ~Tensor_impl() override;

        // public API
        void setName(char const* name) noexcept override;
        char const* getName() const noexcept override;
        void setDimensions(Dims const& dimensions) noexcept override;
        Dims getDimensions() const noexcept override;
        void setType(DataType type) noexcept override;
        DataType getType() const noexcept override;
        bool setDynamicRange(float min, float max) noexcept override;
        bool isNetworkInput() const noexcept override;
        bool isNetworkOutput() const noexcept override;
        void setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept override;
        bool getBroadcastAcrossBatch() const noexcept override;
        TensorLocation getLocation() const noexcept override;
        void setLocation(TensorLocation location) noexcept override;
        bool dynamicRangeIsSet() const noexcept override;
        void resetDynamicRange() noexcept override;
        float getDynamicRangeMin() const noexcept override;
        float getDynamicRangeMax() const noexcept override;
        void setAllowedFormats(TensorFormats formats) noexcept override;
        TensorFormats getAllowedFormats() const noexcept override;
        bool isShapeTensor() const noexcept override;
        bool isExecutionTensor() const noexcept override;
        void setDimensionName(int32_t index, char const* name) noexcept override;
        char const* getDimensionName(int32_t index) const noexcept override;

        migraphx::instruction_ref getInstruction() const noexcept;
        void setInstruction(migraphx::instruction_ref ins) noexcept;

    private:
        migraphx::instruction_ref mIns;
        std::string mName;
        bool mBound = false;
    };
}

#endif // TENSOR_IMPL_HPP
