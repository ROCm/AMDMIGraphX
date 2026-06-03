#ifndef NV_NETWORK_DEFINITION_IMPL_H
#define NV_NETWORK_DEFINITION_IMPL_H

#include <vector>

#include <migraphx/program.hpp>

#include "migraphx/common_api/NvInfer.h"
#include "Tensor_impl.hpp"
#include "Loop_impl.hpp"
#include "layers/Layer_impl.hpp"

namespace nvinfer1
{
    class NvNetworkDefinition_impl : public INetworkDefinition, public apiv::VNetworkDefinition
    {
    public:
        NvNetworkDefinition_impl(NetworkDefinitionCreationFlags flags, IBuilder& builder) noexcept;
        ~NvNetworkDefinition_impl() override;

        // public API
        ITensor* addInput(char const* name, DataType type, Dims const& dimensions) noexcept override;
        void markOutput(ITensor& tensor) noexcept override;
        IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept override;
        ILRNLayer* addLRN(ITensor& input, int64_t window, float alpha, float beta, float k) noexcept override;
        IScaleLayer* addScale(
            ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept override;
        ISoftMaxLayer* addSoftMax(ITensor& input) noexcept override;
        IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept override;
        IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept override;
        IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept override;
        IShuffleLayer* addShuffle(ITensor& input) noexcept override;
        int32_t getNbLayers() const noexcept override;
        ILayer* getLayer(int32_t index) const noexcept override;
        int32_t getNbInputs() const noexcept override;
        ITensor* getInput(int32_t index) const noexcept override;
        int32_t getNbOutputs() const noexcept override;
        ITensor* getOutput(int32_t index) const noexcept override;
        IReduceLayer* addReduce(
            ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) noexcept
            override;
        ITopKLayer* addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept override;
        IGatherLayer* addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept override;
        IRaggedSoftMaxLayer* addRaggedSoftMax(ITensor& input, ITensor& bounds) noexcept override;
        IMatrixMultiplyLayer* addMatrixMultiply(
            ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) noexcept override;
        IConstantLayer* addConstant(Dims const& dimensions, Weights weights) noexcept override;
        IIdentityLayer* addIdentity(ITensor& input) noexcept override;
        void removeTensor(ITensor& tensor) noexcept override;
        void unmarkOutput(ITensor& tensor) noexcept override;
        IPluginV2Layer* addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept override;
        IPluginV3Layer* addPluginV3(ITensor* const* inputs, int32_t nbInputs, ITensor* const* shapeInputs,
            int32_t nbShapeInputs, IPluginV3& plugin) noexcept override;
        ISliceLayer* addSlice(ITensor& input, Dims const& start, Dims const& size, Dims const& stride) noexcept override;
        void setName(char const* name) noexcept override;
        char const* getName() const noexcept override;
        IShapeLayer* addShape(ITensor& input) noexcept override;
        bool hasImplicitBatchDimension() const noexcept override;
        bool markOutputForShapes(ITensor& tensor) noexcept override;
        bool unmarkOutputForShapes(ITensor& tensor) noexcept override;
        IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept override;
        IConvolutionLayer* addConvolutionNd(
            ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
            override;
        IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims const& windowSize) noexcept override;
        IDeconvolutionLayer* addDeconvolutionNd(
            ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
            override;
        IScaleLayer* addScaleNd(
            ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power, int32_t channelAxis) noexcept override;
        IResizeLayer* addResize(ITensor& input) noexcept override;
        ILoop* addLoop() noexcept override;
        ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept override;
        IFillLayer* addFill(Dims const& dimensions, FillOperation op) noexcept override;
        IPaddingLayer* addPaddingNd(ITensor& input, Dims const& prePadding, Dims const& postPadding) noexcept override;
        bool setWeightsName(Weights weights, char const* name) noexcept override;
        void setErrorRecorder(IErrorRecorder* recorder) noexcept override;
        IErrorRecorder* getErrorRecorder() const noexcept override;
        IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept override;
        IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept override;
        IGatherLayer* addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept override;
        IIfConditional* addIfConditional() noexcept override;
        IScatterLayer* addScatter(ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode) noexcept override;
        IEinsumLayer* addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept override;
        IAssertionLayer* addAssertion(ITensor& condition, char const* message) noexcept override;
        IOneHotLayer* addOneHot(ITensor& indices, ITensor& values, ITensor& depth, int32_t axis) noexcept override;
        INonZeroLayer* addNonZero(ITensor& input) noexcept override;
        IGridSampleLayer* addGridSample(ITensor& input, ITensor& grid) noexcept override;
        INMSLayer* addNMS(ITensor& boxes, ITensor& scores, ITensor& maxOutputBoxesPerClass) noexcept override;
        IReverseSequenceLayer* addReverseSequence(ITensor& input, ITensor& sequenceLens) noexcept override;
        INormalizationLayer* addNormalization(
            ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept override;
        ICastLayer* addCast(ITensor& input, DataType toType) noexcept override;
        IBuilder& getBuilder() const noexcept override;
        NetworkDefinitionCreationFlags getFlags() const noexcept override;
        bool getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag) const noexcept override;
        IQuantizeLayer* addQuantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept override;
        IDequantizeLayer* addDequantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept override;
        IFillLayer* addFillV2(Dims const& dimensions, FillOperation op, DataType outputType) noexcept override;
        bool markDebug(ITensor& tensor) noexcept override;
        bool unmarkDebug(ITensor& tensor) noexcept override;
        bool isDebugTensor(ITensor const& tensor) const noexcept override;
        bool markWeightsRefittable(char const* name) noexcept override;
        bool unmarkWeightsRefittable(char const* name) noexcept override;
        bool areWeightsMarkedRefittable(char const* name) const noexcept override;
        ISqueezeLayer* addSqueeze(ITensor& input, ITensor& axes) noexcept override;
        IUnsqueezeLayer* addUnsqueeze(ITensor& input, ITensor& axes) noexcept override;
        IDynamicQuantizeLayer* addDynamicQuantize(
            ITensor& input, int32_t axis, int32_t blockSize, DataType toType, DataType scaleType) noexcept override;
        ICumulativeLayer* addCumulative(
            ITensor& input, ITensor& axis, CumulativeOperation operation, bool exclusive, bool reverse) noexcept override;
        bool markUnfusedTensorsAsDebugTensors() noexcept override;
        bool unmarkUnfusedTensorsAsDebugTensors() noexcept override;

        void setProgram(std::shared_ptr<migraphx::program> program);
        const migraphx::program* getProgram() const;
        
        void build() noexcept;

        // Used by Loop_impl to inspect the regular network layers and to learn
        // the names that marked outputs should be exposed under.
        std::vector<std::unique_ptr<Layer_impl>>& getLayers() noexcept;
        std::vector<std::string> getOutputNames() const;

    private:
        IBuilder& mBuilder;

        std::shared_ptr<migraphx::program> mProgram = std::make_shared<migraphx::program>();
        std::vector<std::unique_ptr<Tensor_impl>> mInputTensors;
        std::vector<Tensor_impl*> mOutputTensors;
        std::vector<std::unique_ptr<Tensor_impl>> mOwnedOutputTensors;
        std::vector<std::unique_ptr<Layer_impl>> mLayers;
        std::vector<std::unique_ptr<Loop_impl>> mLoops;

        // Outputs the caller explicitly bound via markOutput(). These carry
        // user-assigned binding names and are the only outputs whose names the
        // generated parameters should be renamed to. Programs supplied directly
        // (e.g. parsed from ONNX via setProgram) never populate this, so they are
        // left under their native "<module>:#output_N" names.
        std::vector<Tensor_impl*> mMarkedOutputs;
    };

}   // ns:nvinfer1

#endif // NV_NETWORK_DEFINITION_IMPL_H
