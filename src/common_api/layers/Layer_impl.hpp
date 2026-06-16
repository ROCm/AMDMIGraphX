#ifndef LAYER_IMPL_HPP
#define LAYER_IMPL_HPP

#include <vector>
#include <memory>
#include <string>

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

        // Select the migraphx module that build() emits instructions into.
        // When unset (the common case) instructions go into the main module.
        // The loop facade redirects body layers into the loop submodule.
        void setModule(migraphx::module* mod) noexcept;
        migraphx::module* getModule() const noexcept;

        // The explicitly assigned module, or nullptr when the layer targets the
        // main module. Unlike getModule() this does not fall back to main, so the
        // build orchestrator can tell loop-body layers apart from main layers.
        migraphx::module* getAssignedModule() const noexcept { return mModule; }

        // Accessors used by the loop facade to inspect the layer graph.
        const std::vector<Tensor_impl*>& inputTensors() const noexcept { return mInputs; }
        Tensor_impl* outputTensor(int32_t index) const noexcept { return mOutputs.at(index).get(); }

    protected:
        std::vector<migraphx::instruction_ref> getInputArguments() const noexcept;

        LayerType mType;
        std::string mName;
        std::shared_ptr<migraphx::program> mProgram;
        migraphx::module* mModule = nullptr;
        std::vector<Tensor_impl*> mInputs;
        std::vector<std::unique_ptr<Tensor_impl>> mOutputs;
        std::vector<migraphx::instruction_ref> mInstructions;
    };

} // namespace nvinfer1

#endif // LAYER_IMPL_HPP
