#include "NetworkDefinition_impl.hpp"

namespace nvinfer1
{

NvNetworkDefinition_impl::NvNetworkDefinition_impl(NetworkDefinitionCreationFlags flags, IBuilder& builder) noexcept : mBuilder{builder}
{
	// TODO! implement
	mImpl = this;
}

NvNetworkDefinition_impl::~NvNetworkDefinition_impl()
{
	// TODO! implement
}

ITensor* NvNetworkDefinition_impl::addInput(char const* name, DataType type, Dims const& dimensions) noexcept
{
	// TODO! implement
	return nullptr;
}

void NvNetworkDefinition_impl::markOutput(ITensor& tensor) noexcept
{
	// TODO! implement
	return;
}

IActivationLayer* NvNetworkDefinition_impl::addActivation(ITensor& input, ActivationType type) noexcept
{
	// TODO! implement
	return nullptr;
}

ILRNLayer* NvNetworkDefinition_impl::addLRN(ITensor& input, int64_t window, float alpha, float beta, float k) noexcept
{
	// TODO! implement
	return nullptr;
}

IScaleLayer* NvNetworkDefinition_impl::addScale(
    ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept
{
	// TODO! implement
	return nullptr;
}

ISoftMaxLayer* NvNetworkDefinition_impl::addSoftMax(ITensor& input) noexcept
{
	// TODO! implement
	return nullptr;
}

IConcatenationLayer* NvNetworkDefinition_impl::addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept
{
	// TODO! implement
	return nullptr;
}

IElementWiseLayer* NvNetworkDefinition_impl::addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept
{
	// TODO! implement
	return nullptr;
}

IUnaryLayer* NvNetworkDefinition_impl::addUnary(ITensor& input, UnaryOperation operation) noexcept
{
	// TODO! implement
	return nullptr;
}

IShuffleLayer* NvNetworkDefinition_impl::addShuffle(ITensor& input) noexcept
{
	// TODO! implement
	return nullptr;
}

int32_t NvNetworkDefinition_impl::getNbLayers() const noexcept
{
	// TODO! implement
	return 0;
}

ILayer* NvNetworkDefinition_impl::getLayer(int32_t index) const noexcept
{
	// TODO! implement
	return nullptr;
}

int32_t NvNetworkDefinition_impl::getNbInputs() const noexcept
{
	// TODO! implement
	return 0;
}

ITensor* NvNetworkDefinition_impl::getInput(int32_t index) const noexcept
{
	// TODO! implement
	return nullptr;
}

int32_t NvNetworkDefinition_impl::getNbOutputs() const noexcept
{
	// TODO! implement
	return 0;
}

ITensor* NvNetworkDefinition_impl::getOutput(int32_t index) const noexcept
{
	// TODO! implement
	return nullptr;
}

IReduceLayer* NvNetworkDefinition_impl::addReduce(
    ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) noexcept
   
{
	// TODO! implement
	return nullptr;
}

ITopKLayer* NvNetworkDefinition_impl::addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept
{
	// TODO! implement
	return nullptr;
}

IGatherLayer* NvNetworkDefinition_impl::addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept
{
	// TODO! implement
	return nullptr;
}

IRaggedSoftMaxLayer* NvNetworkDefinition_impl::addRaggedSoftMax(ITensor& input, ITensor& bounds) noexcept
{
	// TODO! implement
	return nullptr;
}

IMatrixMultiplyLayer* NvNetworkDefinition_impl::addMatrixMultiply(
    ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) noexcept
{
	// TODO! implement
	return nullptr;
}

IConstantLayer* NvNetworkDefinition_impl::addConstant(Dims const& dimensions, Weights weights) noexcept
{
	// TODO! implement
	return nullptr;
}

IIdentityLayer* NvNetworkDefinition_impl::addIdentity(ITensor& input) noexcept
{
	// TODO! implement
	return nullptr;
}

void NvNetworkDefinition_impl::removeTensor(ITensor& tensor) noexcept
{
	// TODO! implement
	return;
}

void NvNetworkDefinition_impl::unmarkOutput(ITensor& tensor) noexcept
{
	// TODO! implement
	return;
}

IPluginV2Layer* NvNetworkDefinition_impl::addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept
{
	// TODO! implement
	return nullptr;
}

IPluginV3Layer* NvNetworkDefinition_impl::addPluginV3(ITensor* const* inputs, int32_t nbInputs, ITensor* const* shapeInputs,
    int32_t nbShapeInputs, IPluginV3& plugin) noexcept
{
	// TODO! implement
	return nullptr;
}

ISliceLayer* NvNetworkDefinition_impl::addSlice(ITensor& input, Dims const& start, Dims const& size, Dims const& stride) noexcept
{
	// TODO! implement
	return nullptr; 
}

void NvNetworkDefinition_impl::setName(char const* name) noexcept
{
	// TODO! implement
	return;
}

char const* NvNetworkDefinition_impl::getName() const noexcept
{
	// TODO! implement
	return nullptr;
}

IShapeLayer* NvNetworkDefinition_impl::addShape(ITensor& input) noexcept
{
	// TODO! implement
	return nullptr;
}

bool NvNetworkDefinition_impl::hasImplicitBatchDimension() const noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::markOutputForShapes(ITensor& tensor) noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::unmarkOutputForShapes(ITensor& tensor) noexcept
{
	// TODO! implement
	return false;
}

IParametricReLULayer* NvNetworkDefinition_impl::addParametricReLU(ITensor& input, ITensor& slope) noexcept
{
	// TODO! implement
	return nullptr;
}

IConvolutionLayer* NvNetworkDefinition_impl::NvNetworkDefinition_impl::addConvolutionNd(
    ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
   
{
	// TODO! implement
	return nullptr;
}

IPoolingLayer* NvNetworkDefinition_impl::addPoolingNd(ITensor& input, PoolingType type, Dims const& windowSize) noexcept
{
	// TODO! implement
	return nullptr;
}

IDeconvolutionLayer* NvNetworkDefinition_impl::addDeconvolutionNd(
    ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
   
{
	// TODO! implement
	return nullptr;
}

IScaleLayer* NvNetworkDefinition_impl::addScaleNd(
    ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power, int32_t channelAxis) noexcept
{
	// TODO! implement
	return nullptr;
}

IResizeLayer* NvNetworkDefinition_impl::addResize(ITensor& input) noexcept
{
	// TODO! implement
	return nullptr;
}

ILoop* NvNetworkDefinition_impl::addLoop() noexcept
{
	// TODO! implement
	return nullptr;
}

ISelectLayer* NvNetworkDefinition_impl::addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept
{
	// TODO! implement
	return nullptr;
}

IFillLayer* NvNetworkDefinition_impl::addFill(Dims const& dimensions, FillOperation op) noexcept
{
	// TODO! implement
	return nullptr;
}

IPaddingLayer* NvNetworkDefinition_impl::addPaddingNd(ITensor& input, Dims const& prePadding, Dims const& postPadding) noexcept
{
	// TODO! implement
	return nullptr;
}

bool NvNetworkDefinition_impl::setWeightsName(Weights weights, char const* name) noexcept
{
	// TODO! implement
	return false;
}

void NvNetworkDefinition_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
	// TODO! implement
	return;
}

IErrorRecorder* NvNetworkDefinition_impl::getErrorRecorder() const noexcept
{
	// TODO! implement
	return nullptr;
}

IDequantizeLayer* NvNetworkDefinition_impl::addDequantize(ITensor& input, ITensor& scale) noexcept
{
	// TODO! implement
	return nullptr;
}

IQuantizeLayer* NvNetworkDefinition_impl::addQuantize(ITensor& input, ITensor& scale) noexcept
{
	// TODO! implement
	return nullptr;
}

IGatherLayer* NvNetworkDefinition_impl::addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept
{
	// TODO! implement
	return nullptr;
}

IIfConditional* NvNetworkDefinition_impl::addIfConditional() noexcept
{
	// TODO! implement
	return nullptr;
}

IScatterLayer* NvNetworkDefinition_impl::addScatter(ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode) noexcept
{
	// TODO! implement
	return nullptr;
}

IEinsumLayer* NvNetworkDefinition_impl::addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept
{
	// TODO! implement
	return nullptr;
}

IAssertionLayer* NvNetworkDefinition_impl::addAssertion(ITensor& condition, char const* message) noexcept
{
	// TODO! implement
	return nullptr;
}

IOneHotLayer* NvNetworkDefinition_impl::addOneHot(ITensor& indices, ITensor& values, ITensor& depth, int32_t axis) noexcept
{
	// TODO! implement
	return nullptr;
}

INonZeroLayer* NvNetworkDefinition_impl::addNonZero(ITensor& input) noexcept
{
	// TODO! implement
	return nullptr;
}

IGridSampleLayer* NvNetworkDefinition_impl::addGridSample(ITensor& input, ITensor& grid) noexcept
{
	// TODO! implement
	return nullptr;
}

INMSLayer* NvNetworkDefinition_impl::addNMS(ITensor& boxes, ITensor& scores, ITensor& maxOutputBoxesPerClass) noexcept
{
	// TODO! implement
	return nullptr;
}

IReverseSequenceLayer* NvNetworkDefinition_impl::addReverseSequence(ITensor& input, ITensor& sequenceLens) noexcept
{
	// TODO! implement
	return nullptr;
}

INormalizationLayer* NvNetworkDefinition_impl::addNormalization(
    ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept
{
	// TODO! implement
	return nullptr;
}

ICastLayer* NvNetworkDefinition_impl::addCast(ITensor& input, DataType toType) noexcept
{
	// TODO! implement
	return nullptr;
}

IBuilder& NvNetworkDefinition_impl::getBuilder() const noexcept
{
	// TODO! implement
	return mBuilder;
}

NetworkDefinitionCreationFlags NvNetworkDefinition_impl::getFlags() const noexcept
{
	// TODO! implement
	return 0;
}

bool NvNetworkDefinition_impl::getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag) const noexcept
{
	// TODO! implement
	return false;
}

IQuantizeLayer* NvNetworkDefinition_impl::addQuantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept
{
	// TODO! implement
	return nullptr;
}

IDequantizeLayer* NvNetworkDefinition_impl::addDequantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept
{
	// TODO! implement
	return nullptr;
}

IFillLayer* NvNetworkDefinition_impl::addFillV2(Dims const& dimensions, FillOperation op, DataType outputType) noexcept
{
	// TODO! implement
	return nullptr;
}

bool NvNetworkDefinition_impl::markDebug(ITensor& tensor) noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::unmarkDebug(ITensor& tensor) noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::isDebugTensor(ITensor const& tensor) const noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::markWeightsRefittable(char const* name) noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::unmarkWeightsRefittable(char const* name) noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::areWeightsMarkedRefittable(char const* name) const noexcept
{
	// TODO! implement
	return false;
}

ISqueezeLayer* NvNetworkDefinition_impl::addSqueeze(ITensor& input, ITensor& axes) noexcept
{
	// TODO! implement
	return nullptr;
}

IUnsqueezeLayer* NvNetworkDefinition_impl::addUnsqueeze(ITensor& input, ITensor& axes) noexcept
{
	// TODO! implement
	return nullptr;
}

IDynamicQuantizeLayer* NvNetworkDefinition_impl::addDynamicQuantize(
    ITensor& input, int32_t axis, int32_t blockSize, DataType toType, DataType scaleType) noexcept
{
	// TODO! implement
	return nullptr;
}

ICumulativeLayer* NvNetworkDefinition_impl::addCumulative(
    ITensor& input, ITensor& axis, CumulativeOperation operation, bool exclusive, bool reverse) noexcept
{
	// TODO! implement
	return nullptr;
}

bool NvNetworkDefinition_impl::markUnfusedTensorsAsDebugTensors() noexcept
{
	// TODO! implement
	return false;
}

bool NvNetworkDefinition_impl::unmarkUnfusedTensorsAsDebugTensors() noexcept
{
	// TODO! implement
	return false;
}



}  // namespace nvinfer1
