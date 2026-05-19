#ifndef LAYER_IMPL_HPP
#define LAYER_IMPL_HPP

#include <vector>
#include <memory>

#include <migraphx/program.hpp>

#include "migraphx/common_api/NvInfer.h"
#include "Tensor_impl.hpp"

namespace nvinfer1
{
    class Layer_impl : public ILayer, public apiv::VLayer
    {
    public:
        Layer_impl() noexcept;
        Layer_impl(LayerType type, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~Layer_impl() override;
    
        // public API
        LayerType getType() const noexcept override;
        void setName(char const* name) noexcept override;
        char const* getName() const noexcept override;
        int32_t getNbInputs() const noexcept override;
        ITensor* getInput(int32_t index) const noexcept override;
        int32_t getNbOutputs() const noexcept override;
        ITensor* getOutput(int32_t index) const noexcept override;
        void setInput(int32_t index, ITensor& tensor) noexcept override;
        void setPrecision(DataType dataType) noexcept override;
        DataType getPrecision() const noexcept override;
        bool precisionIsSet() const noexcept override;
        void resetPrecision() noexcept override;
        void setOutputType(int32_t index, DataType dataType) noexcept override;
        DataType getOutputType(int32_t index) const noexcept override;
        bool outputTypeIsSet(int32_t index) const noexcept override;
        void resetOutputType(int32_t index) noexcept override;
        void setMetadata(char const* docString) noexcept override;
        char const* getMetadata() const noexcept override;

        virtual void build() noexcept = 0;

    protected:
        std::vector<migraphx::instruction_ref> getInputArguments() const noexcept;

        LayerType mType;
        std::shared_ptr<migraphx::program> mProgram;
        std::vector<Tensor_impl*> mInputs;
        std::vector<std::unique_ptr<Tensor_impl>> mOutputs;
        std::vector<migraphx::instruction_ref> mInstructions;
    };

} // namespace nvinfer1

#endif // LAYER_IMPL_HPP
