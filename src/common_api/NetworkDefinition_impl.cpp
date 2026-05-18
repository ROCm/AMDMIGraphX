// TODO! remove when all methods are implemented
#include "pass_warning.hpp"
//

#include "Helper.hpp"
#include "NetworkDefinition_impl.hpp"

// layers
#include "layers/ConstantLayer_impl.hpp"

namespace nvinfer1
{

NvNetworkDefinition_impl::NvNetworkDefinition_impl(NetworkDefinitionCreationFlags flags, IBuilder& builder) noexcept : mBuilder{builder}
{
	pass_warning("TODO! implement me!", false);
	mImpl = this;
}

NvNetworkDefinition_impl::~NvNetworkDefinition_impl()
{
	pass_warning("TODO! implement me!", false);
}

const migraphx::program* NvNetworkDefinition_impl::getProgram() const
{
	return mProgram.get(); 
}

void NvNetworkDefinition_impl::setProgram(std::shared_ptr<migraphx::program> program)
{
	mProgram = program;

	for(auto param : mProgram->get_main_module()->get_parameters())
	{
		mInputTensors.push_back(std::make_unique<Tensor_impl>(param));
	}

	for(auto param : mProgram->get_main_module()->get_returns())
	{
		mOwnedOutputTensors.push_back(std::make_unique<Tensor_impl>(param));
		mOutputTensors.push_back(mOwnedOutputTensors.back().get());
	}
}

ITensor* NvNetworkDefinition_impl::addInput(char const* name, DataType type, Dims const& dimensions) noexcept
{
	auto* mm = mProgram->get_main_module();
	auto input = mm->add_parameter(name, migraphx::shape{helper::fromDataType(type), helper::dimsToVec(dimensions)});
	mInputTensors.push_back(std::make_unique<Tensor_impl>(input));
	auto* ret = mInputTensors.back().get();
	return ret;
}

void NvNetworkDefinition_impl::markOutput(ITensor& tensor) noexcept
{
	Tensor_impl* impl = dynamic_cast<Tensor_impl*>(&tensor);
	mOutputTensors.push_back(impl);
	return;
}

IActivationLayer* NvNetworkDefinition_impl::addActivation(ITensor& input, ActivationType type) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ILRNLayer* NvNetworkDefinition_impl::addLRN(ITensor& input, int64_t window, float alpha, float beta, float k) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IScaleLayer* NvNetworkDefinition_impl::addScale(
    ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ISoftMaxLayer* NvNetworkDefinition_impl::addSoftMax(ITensor& input) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IConcatenationLayer* NvNetworkDefinition_impl::addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IElementWiseLayer* NvNetworkDefinition_impl::addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IUnaryLayer* NvNetworkDefinition_impl::addUnary(ITensor& input, UnaryOperation operation) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IShuffleLayer* NvNetworkDefinition_impl::addShuffle(ITensor& input) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

int32_t NvNetworkDefinition_impl::getNbLayers() const noexcept
{
	pass_warning("TODO! implement me!", true);
	return 0;
}

ILayer* NvNetworkDefinition_impl::getLayer(int32_t index) const noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

int32_t NvNetworkDefinition_impl::getNbInputs() const noexcept
{
	return mInputTensors.size();
}

ITensor* NvNetworkDefinition_impl::getInput(int32_t index) const noexcept
{
	return mInputTensors.at(index).get();
}

int32_t NvNetworkDefinition_impl::getNbOutputs() const noexcept
{
	return mOutputTensors.size();
}

ITensor* NvNetworkDefinition_impl::getOutput(int32_t index) const noexcept
{
	return mOutputTensors.at(index);
}

IReduceLayer* NvNetworkDefinition_impl::addReduce(
    ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) noexcept
   
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ITopKLayer* NvNetworkDefinition_impl::addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IGatherLayer* NvNetworkDefinition_impl::addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IRaggedSoftMaxLayer* NvNetworkDefinition_impl::addRaggedSoftMax(ITensor& input, ITensor& bounds) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IMatrixMultiplyLayer* NvNetworkDefinition_impl::addMatrixMultiply(
    ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IConstantLayer* NvNetworkDefinition_impl::addConstant(Dims const& dimensions, Weights weights) noexcept
{
	mLayers.push_back(std::make_unique<ConstantLayer_impl>(dimensions, weights, mProgram));
    return dynamic_cast<IConstantLayer*>(mLayers.back().get());
}

IIdentityLayer* NvNetworkDefinition_impl::addIdentity(ITensor& input) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

void NvNetworkDefinition_impl::removeTensor(ITensor& tensor) noexcept
{
	pass_warning("TODO! implement me!", true);
	return;
}

void NvNetworkDefinition_impl::unmarkOutput(ITensor& tensor) noexcept
{
	pass_warning("TODO! implement me!", true);
	return;
}

IPluginV2Layer* NvNetworkDefinition_impl::addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IPluginV3Layer* NvNetworkDefinition_impl::addPluginV3(ITensor* const* inputs, int32_t nbInputs, ITensor* const* shapeInputs,
    int32_t nbShapeInputs, IPluginV3& plugin) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ISliceLayer* NvNetworkDefinition_impl::addSlice(ITensor& input, Dims const& start, Dims const& size, Dims const& stride) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr; 
}

void NvNetworkDefinition_impl::setName(char const* name) noexcept
{
	pass_warning("TODO! implement me!", true);
	return;
}

char const* NvNetworkDefinition_impl::getName() const noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IShapeLayer* NvNetworkDefinition_impl::addShape(ITensor& input) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

bool NvNetworkDefinition_impl::hasImplicitBatchDimension() const noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::markOutputForShapes(ITensor& tensor) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::unmarkOutputForShapes(ITensor& tensor) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

IParametricReLULayer* NvNetworkDefinition_impl::addParametricReLU(ITensor& input, ITensor& slope) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IConvolutionLayer* NvNetworkDefinition_impl::NvNetworkDefinition_impl::addConvolutionNd(
    ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
   
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IPoolingLayer* NvNetworkDefinition_impl::addPoolingNd(ITensor& input, PoolingType type, Dims const& windowSize) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IDeconvolutionLayer* NvNetworkDefinition_impl::addDeconvolutionNd(
    ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
   
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IScaleLayer* NvNetworkDefinition_impl::addScaleNd(
    ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power, int32_t channelAxis) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IResizeLayer* NvNetworkDefinition_impl::addResize(ITensor& input) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ILoop* NvNetworkDefinition_impl::addLoop() noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ISelectLayer* NvNetworkDefinition_impl::addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IFillLayer* NvNetworkDefinition_impl::addFill(Dims const& dimensions, FillOperation op) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IPaddingLayer* NvNetworkDefinition_impl::addPaddingNd(ITensor& input, Dims const& prePadding, Dims const& postPadding) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

bool NvNetworkDefinition_impl::setWeightsName(Weights weights, char const* name) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

void NvNetworkDefinition_impl::setErrorRecorder(IErrorRecorder* recorder) noexcept
{
	pass_warning("TODO! implement me!", true);
	return;
}

IErrorRecorder* NvNetworkDefinition_impl::getErrorRecorder() const noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IDequantizeLayer* NvNetworkDefinition_impl::addDequantize(ITensor& input, ITensor& scale) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IQuantizeLayer* NvNetworkDefinition_impl::addQuantize(ITensor& input, ITensor& scale) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IGatherLayer* NvNetworkDefinition_impl::addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IIfConditional* NvNetworkDefinition_impl::addIfConditional() noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IScatterLayer* NvNetworkDefinition_impl::addScatter(ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IEinsumLayer* NvNetworkDefinition_impl::addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IAssertionLayer* NvNetworkDefinition_impl::addAssertion(ITensor& condition, char const* message) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IOneHotLayer* NvNetworkDefinition_impl::addOneHot(ITensor& indices, ITensor& values, ITensor& depth, int32_t axis) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

INonZeroLayer* NvNetworkDefinition_impl::addNonZero(ITensor& input) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IGridSampleLayer* NvNetworkDefinition_impl::addGridSample(ITensor& input, ITensor& grid) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

INMSLayer* NvNetworkDefinition_impl::addNMS(ITensor& boxes, ITensor& scores, ITensor& maxOutputBoxesPerClass) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IReverseSequenceLayer* NvNetworkDefinition_impl::addReverseSequence(ITensor& input, ITensor& sequenceLens) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

INormalizationLayer* NvNetworkDefinition_impl::addNormalization(
    ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ICastLayer* NvNetworkDefinition_impl::addCast(ITensor& input, DataType toType) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IBuilder& NvNetworkDefinition_impl::getBuilder() const noexcept
{
	pass_warning("TODO! implement me!", true);
	return mBuilder;
}

NetworkDefinitionCreationFlags NvNetworkDefinition_impl::getFlags() const noexcept
{
	pass_warning("TODO! implement me!", true);
	return 0;
}

bool NvNetworkDefinition_impl::getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag) const noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

IQuantizeLayer* NvNetworkDefinition_impl::addQuantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IDequantizeLayer* NvNetworkDefinition_impl::addDequantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IFillLayer* NvNetworkDefinition_impl::addFillV2(Dims const& dimensions, FillOperation op, DataType outputType) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

bool NvNetworkDefinition_impl::markDebug(ITensor& tensor) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::unmarkDebug(ITensor& tensor) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::isDebugTensor(ITensor const& tensor) const noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::markWeightsRefittable(char const* name) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::unmarkWeightsRefittable(char const* name) noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::areWeightsMarkedRefittable(char const* name) const noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

ISqueezeLayer* NvNetworkDefinition_impl::addSqueeze(ITensor& input, ITensor& axes) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IUnsqueezeLayer* NvNetworkDefinition_impl::addUnsqueeze(ITensor& input, ITensor& axes) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

IDynamicQuantizeLayer* NvNetworkDefinition_impl::addDynamicQuantize(
    ITensor& input, int32_t axis, int32_t blockSize, DataType toType, DataType scaleType) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

ICumulativeLayer* NvNetworkDefinition_impl::addCumulative(
    ITensor& input, ITensor& axis, CumulativeOperation operation, bool exclusive, bool reverse) noexcept
{
	pass_warning("TODO! implement me!", true);
	return nullptr;
}

bool NvNetworkDefinition_impl::markUnfusedTensorsAsDebugTensors() noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

bool NvNetworkDefinition_impl::unmarkUnfusedTensorsAsDebugTensors() noexcept
{
	pass_warning("TODO! implement me!", true);
	return false;
}

void NvNetworkDefinition_impl::build() noexcept
{
	std::for_each(mLayers.begin(), mLayers.end(),
		[](auto& layer)
		{
			layer->build();
		}
	);
}

}  // namespace nvinfer1
