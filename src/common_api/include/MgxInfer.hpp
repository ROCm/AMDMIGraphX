#pragma once

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <migraphx/onnx.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include "MgxInferRuntimeBase.hpp"
#include "migraphx/migraphx.h"
#include "pass.hpp"

namespace mgxinfer1 {

// TODO interfaces
class IGpuAllocator
{
};
class IErrorRecorder
{
};
class IStreamReader
{
};
class IPluginRegistry
{
};
class IEngineInspector
{
};
class ISerializationConfig
{
};
class IProfiler
{
};
class IOutputAllocator
{
};
class IDebugListener
{
};
class IOptimizationProfile
{
};
class IActivationLayer
{
};
class ILRNLayer
{
};
class IScaleLayer
{
};
class ISoftMaxLayer
{
};
class IConcatenationLayer
{
};
class IShuffleLayer
{
};
class IOneHotLayer
{
};
class ITopKLayer
{
};
class IGatherLayer
{
};
class IRaggedSoftMaxLayer
{
};
class IMatrixMultiplyLayer
{
};
class INonZeroLayer
{
};
class IConstantLayer
{
};
class IIdentityLayer
{
};
class ICastLayer
{
};
class IPluginV2Layer
{
};
class IPluginV3Layer
{
};
class ISliceLayer
{
};
class IShapeLayer
{
};
class IParametricReLULayer
{
};
class IConvolutionLayer
{
};
class IPoolingLayer
{
};
class IDeconvolutionLayer
{
};
class IResizeLayer
{
};
class ILoop
{
};
class IIfConditional
{
};
class ISelectLayer
{
};
class IAssertionLayer
{
};
class IFillLayer
{
};
class IPaddingLayer
{
};
class IDequantizeLayer
{
};
class IScatterLayer
{
};
class IQuantizeLayer
{
};
class IEinsumLayer
{
};
class IGridSampleLayer
{
};
class INMSLayer
{
};
class IReverseSequenceLayer
{
};
class INormalizationLayer
{
};
class IPluginV2
{
};
class IPluginV3
{
};
class IInt8Calibrator
{
};
class IAlgorithmSelector
{
};
class ITimingCache
{
};
class IProgressMonitor
{
};

class ICudaEngine;
class IExecutionContext;
class INetworkDefinition;
class IBuilder;
class IBuilderConfig;

using TensorFormats = uint32_t;
using BuilderFlags  = uint32_t;

//!
//! \class INoCopy
//!
//! \brief Base class for all TensorRT interfaces that are implemented by the TensorRT libraries
//!
//! Objects of such classes are not movable or copyable, and should only be manipulated
//! via pointers.
//!
class INoCopy
{
    protected:
    INoCopy()                                = default;
    virtual ~INoCopy()                       = default;
    INoCopy(INoCopy const& other)            = delete;
    INoCopy& operator=(INoCopy const& other) = delete;
    INoCopy(INoCopy&& other)                 = delete;
    INoCopy& operator=(INoCopy&& other)      = delete;
};

//!
//! \class ITensor
//!
//! \brief A tensor in a network definition.
//!
//! To remove a tensor from a network definition, use INetworkDefinition::removeTensor().
//!
//! When using the DLA, the cumulative size of all Tensors that are not marked as Network Input or
//! Output tensors, must be less than 1GB in size to fit into a single subgraph. If the build option
//! kGPU_FALLBACK is specified, then multiple subgraphs can be created, with each subgraph limited
//! to less than 1GB of internal tensors data.
//!
//! \warning The volume of the tensor must be less than 2^31 elements. If the tensor is a shape
//! tensor, its volume must not exceed 64. \warning Do not inherit from this class, as doing so will
//! break forward-compatibility of the API and ABI.
//!
class ITensor : public INoCopy
{
    public:
    ITensor(migraphx::instruction_ref ins) : ins_{ins} {}
    //!
    //! \brief Set the tensor name.
    //!
    //! For a network input, the name is assigned by the application. For tensors which are layer
    //! outputs, a default name is assigned consisting of the layer name followed by the index of
    //! the output in brackets.
    //!
    //! This method copies the name string.
    //!
    //! \param name The name.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setName(name);
    }

    //!
    //! \brief Get the tensor name.
    //!
    //! \return The name as a null-terminated C-style string.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getName();
    }

    //!
    //! \brief Set the dimensions of a tensor.
    //!
    //! For a network input, the dimensions are assigned by the application. For a network output,
    //! the dimensions are computed based on the layer parameters and the inputs to the layer. If a
    //! tensor size or a parameter is modified in the network, the dimensions of all dependent
    //! tensors will be recomputed.
    //!
    //! This call is only legal for network input tensors, since the dimensions of layer output
    //! tensors are inferred based on layer inputs and parameters. The volume must be less than 2^31
    //! elements.
    //!
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \see getDimensions()
    //!
    void setDimensions(Dims const& dimensions) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDimensions(dimensions);
    }

    //!
    //! \brief Get the dimensions of a tensor.
    //!
    //! \return The dimensions of the tensor.
    //!
    //! \warning getDimensions() returns a -1 for dimensions that are derived from a wildcard
    //! dimension.
    //!
    //! \see setDimensions()
    //!
    Dims getDimensions() const noexcept
    {
        // TODO incomplete
        return toDimensions(ins_->get_shape());
        // pass("Not Implemented", true);
        // return mImpl->getDimensions();
    }

    //!
    //! \brief Set the data type of a tensor.
    //!
    //! \param type The data type of the tensor.
    //!
    //! The type is unchanged if the tensor is not a network input tensor, or marked as an output
    //! tensor or shape output tensor.
    //!
    //! \see getType()
    //!
    void setType(DataType type) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setType(type);
    }

    //!
    //! \brief Get the data type of a tensor.
    //!
    //! \return The data type of the tensor.
    //!
    //! \see setType()
    //!
    DataType getType() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getType();
    }

    //!
    //! \brief Set dynamic range for the tensor
    //!
    //! Currently, only symmetric ranges are supported.
    //! Therefore, the larger of the absolute values of the provided bounds is used.
    //!
    //! \return Whether the dynamic range was set successfully.
    //!
    //! Requires that min and max be finite, and min <= max.
    //!
    bool setDynamicRange(float min, float max) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setDynamicRange(min, max);
    }

    //!
    //! \brief Whether the tensor is a network input.
    //!
    bool isNetworkInput() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isNetworkInput();
    }

    //!
    //! \brief Whether the tensor is a network output.
    //!
    bool isNetworkOutput() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isNetworkOutput();
    }

    //!
    //! \brief Set whether to enable broadcast of tensor across the implicit batch dimension.
    //!
    //! \warning This method has no effect other than issuing a warning.
    //!
    //! \param broadcastAcrossBatch Whether to broadcast the tensor across the implicit
    //!         batch dimension that was a feature of TensorRT 9.x and prior.
    //!
    //! \see getBroadcastAcrossBatch()
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is not supported since
    //! TensorRT 10.0.
    //!
    [[deprecated]] void setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setBroadcastAcrossBatch(broadcastAcrossBatch);
    }

    //!
    //! \brief Check if tensor is broadcast across the implicit batch dimension.
    //!
    //! \return Always false since TensorRT 10.0 does not support an implicit batch dimension.
    //!
    //! \see setBroadcastAcrossBatch()
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is not supported since
    //! TensorRT 10.0.
    //!
    [[deprecated]] bool getBroadcastAcrossBatch() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getBroadcastAcrossBatch();
    }

    //!
    //! \brief Get the storage location of a tensor.
    //!
    //! \return The location of tensor data.
    //!
    //! \see setLocation()
    //!
    TensorLocation getLocation() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getLocation();
    }

    //!
    //! \brief Set the storage location of a tensor
    //!
    //! \param location the location of tensor data
    //!
    //! Only network input tensors for storing sequence lengths for RNNv2 are supported.
    //! Using host storage for layers that do not support it will generate
    //! errors at build time.
    //!
    //! \see getLocation()
    //!
    //! \deprecated Deprecated in TensorRT 10.0. RNNv2 is not supported and the location must
    //! always be TensorLocation::kDEVICE since TensorRT 10.0.
    //!
    [[deprecated]] void setLocation(TensorLocation location) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setLocation(location);
    }

    //!
    //! \brief Query whether dynamic range is set.
    //!
    //! \return True if dynamic range is set, false otherwise.
    //!
    bool dynamicRangeIsSet() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->dynamicRangeIsSet();
    }

    //!
    //! \brief Undo effect of setDynamicRange.
    //!
    void resetDynamicRange() noexcept
    {
        pass("Not Implemented", true);
        // mImpl->resetDynamicRange();
    }

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range, or quiet NaN if range was not set.
    //!
    float getDynamicRangeMin() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDynamicRangeMin();
    }

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range, or quiet NaN if range was not set.
    //!
    float getDynamicRangeMax() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDynamicRangeMax();
    }

    //!
    //! \brief Set allowed formats for this tensor. By default all formats are allowed.
    //!        Shape tensors (for which isShapeTensor() returns true) may only have row-major linear
    //!        format.
    //!
    //! When running network on DLA and the build option kGPU_FALLBACK is not specified, if DLA
    //! format(kCHW4 with Int8, kCHW4 with FP16, kCHW16 with FP16, kCHW32 with Int8) is set, the
    //! input format is treated as native DLA format with line stride requirement. Input/output
    //! binding with these format should have correct layout during inference.
    //!
    //! \param formats A bitmask of TensorFormat values that are supported for this tensor.
    //!
    //! \see ITensor::getAllowedFormats()
    //!
    //! \see TensorFormats
    //!
    void setAllowedFormats(TensorFormats formats) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setAllowedFormats(formats);
    }

    //!
    //! \brief Get a bitmask of TensorFormat values that the tensor supports.
    //!        For a shape tensor, only row-major linear format is allowed.
    //!
    //! \return The value specified by setAllowedFormats or all possible formats.
    //!
    //! \see ITensor::setAllowedFormats()
    //!
    TensorFormats getAllowedFormats() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getAllowedFormats();
    }

    //!
    //! \brief Whether the tensor is a shape tensor.
    //!
    //! A shape tensor is a tensor that is related to shape calculations.
    //! It must have type Int32, Int64, Bool, or Float, and its shape must be determinable at build
    //! time. Furthermore, it must be needed as a shape tensor, either marked as a network shape
    //! output via markOutputForShapes(), or as a layer input that is required to be a shape
    //! tensor, such as the second input to IShuffleLayer. Some layers are "polymorphic" in
    //! this respect. For example, the inputs to IElementWiseLayer must be shape tensors
    //! if the output is a shape tensor.
    //!
    //! The TensorRT Developer Guide give the formal rules for what tensors are shape tensors.
    //!
    //! The result of isShapeTensor() is reliable only when network construction is complete.
    //! For example, if a partially built network sums two tensors T1 and T2 to create
    //! tensor T3, and none are yet needed as shape tensors, isShapeTensor() returns false
    //! for all three tensors.  Setting the second input of IShuffleLayer to be T3 would
    //! cause all three tensors to be shape tensors, because IShuffleLayer requires that its
    //! second optional input be a shape tensor, and IElementWiseLayer is "polymorphic".
    //!
    //! It is possible for a tensor to be both a shape tensor and an execution tensor.
    //!
    //! \return True if tensor is a shape tensor, false otherwise.
    //!
    //! \see INetworkDefinition::markOutputForShapes()
    //!
    bool isShapeTensor() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isShapeTensor();
    }

    //!
    //! \brief Whether the tensor is an execution tensor.
    //!
    //! Tensors are usually execution tensors.  The exceptions are tensors used
    //! solely for shape calculations or whose contents not needed to compute the outputs.
    //!
    //! The result of isExecutionTensor() is reliable only when network construction is complete.
    //! For example, if a partially built network has no path from a tensor to a network output,
    //! isExecutionTensor() returns false. Completing the path would cause it to become true.
    //!
    //!
    //! A tensor with isShapeTensor() == false and isExecutionTensor() == false
    //! can still show up as an input to the engine if its dimensions are required.
    //! In that case, only its dimensions need to be set at runtime and a nullptr
    //! can be passed instead of a pointer to its contents.
    //!
    bool isExecutionTensor() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isExecutionTensor();
    }

    //!
    //! \brief Name a dimension of an input tensor.
    //!
    //! Associate a runtime dimension of an input tensor with a symbolic name.
    //! Dimensions with the same non-empty name must be equal at runtime.
    //! Knowing this equality for runtime dimensions may help the TensorRT optimizer.
    //! Both runtime and build-time dimensions can be named.
    //!
    //! For example, setDimensionName(0, "n") associates the symbolic name "n" with the leading
    //! dimension.
    //!
    //! This method copies the name string.
    //! If the function is called again, with the same index, it will overwrite the previous name.
    //! If nullptr is passed as name, it will clear the name of the dimension.
    //!
    //! \param index index of the dimension
    //! \param name of the dimension, as a pointer to a null-terminated character sequence.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see getDimensionName()
    //!
    void setDimensionName(int32_t index, char const* name) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDimensionName(index, name);
    }

    //!
    //! \brief Get the name of an input dimension.
    //!
    //! \param index index of the dimension
    //!
    //! \return The name of the input dimension, or nullptr if the dimension has no name.
    //!         The name is a pointer to a null-terminated character sequence.
    //!
    //! \see setDimensionName()
    //!
    char const* getDimensionName(int32_t index) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDimensionName(index);
    }

    migraphx::instruction_ref getInstruction() const noexcept { return ins_; }

    // TODO this destructor was protected. For now it is moved to public to avoid compile errors.
    virtual ~ITensor() noexcept = default;

    protected:
    // apiv::VTensor* mImpl;

    private:
    migraphx::instruction_ref ins_;
};

//!
//! \enum LayerType
//!
//! \brief The type values of layer classes.
//!
//! \see ILayer::getType()
//!
enum class LayerType : int32_t
{
    kCONVOLUTION        = 0,  //!< Convolution layer.
    kCAST               = 1,  //!< Cast layer
    kACTIVATION         = 2,  //!< Activation layer.
    kPOOLING            = 3,  //!< Pooling layer.
    kLRN                = 4,  //!< LRN layer.
    kSCALE              = 5,  //!< Scale layer.
    kSOFTMAX            = 6,  //!< SoftMax layer.
    kDECONVOLUTION      = 7,  //!< Deconvolution layer.
    kCONCATENATION      = 8,  //!< Concatenation layer.
    kELEMENTWISE        = 9,  //!< Elementwise layer.
    kPLUGIN             = 10, //!< Plugin layer.
    kUNARY              = 11, //!< UnaryOp operation Layer.
    kPADDING            = 12, //!< Padding layer.
    kSHUFFLE            = 13, //!< Shuffle layer.
    kREDUCE             = 14, //!< Reduce layer.
    kTOPK               = 15, //!< TopK layer.
    kGATHER             = 16, //!< Gather layer.
    kMATRIX_MULTIPLY    = 17, //!< Matrix multiply layer.
    kRAGGED_SOFTMAX     = 18, //!< Ragged softmax layer.
    kCONSTANT           = 19, //!< Constant layer.
    kIDENTITY           = 20, //!< Identity layer.
    kPLUGIN_V2          = 21, //!< PluginV2 layer.
    kSLICE              = 22, //!< Slice layer.
    kSHAPE              = 23, //!< Shape layer.
    kPARAMETRIC_RELU    = 24, //!< Parametric ReLU layer.
    kRESIZE             = 25, //!< Resize Layer.
    kTRIP_LIMIT         = 26, //!< Loop Trip limit layer
    kRECURRENCE         = 27, //!< Loop Recurrence layer
    kITERATOR           = 28, //!< Loop Iterator layer
    kLOOP_OUTPUT        = 29, //!< Loop output layer
    kSELECT             = 30, //!< Select layer.
    kFILL               = 31, //!< Fill layer
    kQUANTIZE           = 32, //!< Quantize layer
    kDEQUANTIZE         = 33, //!< Dequantize layer
    kCONDITION          = 34, //!< Condition layer
    kCONDITIONAL_INPUT  = 35, //!< Conditional Input layer
    kCONDITIONAL_OUTPUT = 36, //!< Conditional Output layer
    kSCATTER            = 37, //!< Scatter layer
    kEINSUM             = 38, //!< Einsum layer
    kASSERTION          = 39, //!< Assertion layer
    kONE_HOT            = 40, //!< OneHot layer
    kNON_ZERO           = 41, //!< NonZero layer
    kGRID_SAMPLE        = 42, //!< Grid sample layer
    kNMS                = 43, //!< NMS layer
    kREVERSE_SEQUENCE   = 44, //!< Reverse sequence layer
    kNORMALIZATION      = 45, //!< Normalization layer
    kPLUGIN_V3          = 46  //!< PluginV3 layer.
};

//!
//! \class ILayer
//!
//! \brief Base class for all layer classes in a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class ILayer : public INoCopy
{
    public:
    ILayer(LayerType type, const std::shared_ptr<migraphx::program>& program)
        : type_{type}, program_{std::move(program)}
    {
    }

    //!
    //! \brief Return the type of a layer.
    //!
    //! \see LayerType
    //!
    LayerType getType() const noexcept
    {
        return type_;
        // return mLayer->getType();
    }

    //!
    //! \brief Set the name of a layer.
    //!
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        pass("Not Implemented", true);
        // mLayer->setName(name);
    }

    //!
    //! \brief Return the name of a layer.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->getName();
    }

    //!
    //! \brief Get the number of inputs of a layer.
    //!
    int32_t getNbInputs() const noexcept
    {
        return inputs_.size();
        // return mLayer->getNbInputs();
    }

    //!
    //! \brief Get the layer input corresponding to the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range or the tensor is optional
    //! (\ref ISliceLayer).
    //!
    ITensor* getInput(int32_t index) const noexcept
    {
        // TODO Index checking
        return inputs_.at(index);
        // return mLayer->getInput(index);
    }

    //!
    //! \brief Get the number of outputs of a layer.
    //!
    int32_t getNbOutputs() const noexcept
    {
        return outputs_.size();
        // return mLayer->getNbOutputs();
    }

    //!
    //! \brief Get the layer output corresponding to the given index.
    //!
    //! \return The indexed output tensor, or nullptr if the index is out of range or the tensor is
    //! optional.
    //!
    ITensor* getOutput(int32_t index) const noexcept
    {
        // TODO add index checking
        return outputs_.at(index).get();
        // return mLayer->getOutput(index);
    }

    //!
    //! \brief Replace an input of this layer with a specific tensor.
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Except for IFillLayer, ILoopOutputLayer, INMSLayer, IResizeLayer, IShuffleLayer, and
    //! ISliceLayer, this method cannot change the number of inputs to a layer. The index argument
    //! must be less than the value of getNbInputs().
    //!
    //! See comments for overloads of setInput() for layers with special behavior.
    //!
    void setInput(int32_t index, ITensor& tensor) noexcept
    {
        // TODO add index checking
        // TODO check if new input shape is different?
        auto* old_input = inputs_.at(index);
        inputs_[index]  = &tensor;
        migraphx::instruction::replace_argument(
            first_ins_, old_input->getInstruction(), tensor.getInstruction());
        // return mLayer->setInput(index, tensor);
    }

    //!
    //! \brief Set the preferred or required computational precision of this layer in a weakly-typed
    //! network.
    //!
    //! Setting the precision directs TensorRT to choose an implementation that runs at this
    //! computational precision. TensorRT could still choose a non-conforming fastest implementation
    //! that ignores the requested precision. To force choosing an implementation with the requested
    //! precision, set exactly one of the following flags, which differ in what happens if no such
    //! implementation exists:
    //!
    //! * BuilderFlag::kOBEY_PRECISION_CONSTRAINTS - build fails with an error message.
    //!
    //! * BuilderFlag::kPREFER_PRECISION_CONSTRAINTS - TensorRT falls back to an
    //!   implementation without the requested precision.
    //!
    //! If precision is not set, or falling back, TensorRT will select the layer computational
    //! precision and layer input type based on global performance considerations and the flags
    //! specified to the builder.
    //!
    //! For a IIdentityLayer: If it casts to/from float/half/int8/uint8, the precision must be one
    //! of those types, otherwise it must be either the input or output type.
    //!
    //! Strongly-typed networks reject calls to method setPrecision. In strongly-typed networks, the
    //! computation precision is typically controlled by casting the input tensors to the desired
    //! type. The exception is INormalizationLayer, which has a method setComputePrecision().
    //!
    //! \param dataType the computational precision.
    //!
    //! \see getPrecision() precisionIsSet() resetPrecision()
    //!
    void setPrecision(DataType dataType) noexcept
    {
        pass("Not Implemented", true);
        // mLayer->setPrecision(dataType);
    }

    //!
    //! \brief get the computational precision of this layer
    //!
    //! \return the computational precision
    //!
    //! \see setPrecision() precisionIsSet() resetPrecision()
    //!
    DataType getPrecision() const noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->getPrecision();
    }

    //!
    //! \brief whether the computational precision has been set for this layer
    //!
    //! \return whether the computational precision has been explicitly set
    //!
    //! \see setPrecision() getPrecision() resetPrecision()
    //!
    bool precisionIsSet() const noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->precisionIsSet();
    }

    //!
    //! \brief reset the computational precision for this layer
    //!
    //! \see setPrecision() getPrecision() precisionIsSet()
    //!
    void resetPrecision() noexcept
    {
        pass("Not Implemented", true);
        // mLayer->resetPrecision();
    }

    //!
    //! \brief Set the output type of this layer in a weakly-typed network.
    //!
    //! Setting the output type constrains TensorRT to choose implementations which generate output
    //! data with the given type. If it is not set, TensorRT will select output type based on layer
    //! computational precision. TensorRT could still choose non-conforming output type based on
    //! fastest implementation. To force choosing the requested output type, set exactly one of the
    //! following flags, which differ in what happens if no such implementation exists:
    //!
    //! * BuilderFlag::kOBEY_PRECISION_CONSTRAINTS - build fails with an error message.
    //!
    //! * BuilderFlag::kPREFER_PRECISION_CONSTRAINTS - TensorRT falls back to an
    //!   implementation with a non-conforming output type.
    //!
    //! In case layer precision is not specified, or falling back, the output type depends on the
    //! chosen implementation, based on performance considerations and the flags specified to the
    //! builder.
    //!
    //! This method cannot be used to set the data type of the second output tensor of the TopK
    //! layer. The data type of the second output tensor of the topK layer is always Int32. Also the
    //! output type of all layers that are shape operations must be DataType::kINT32, and all
    //! attempts to set the output type to some other data type will be ignored except for issuing
    //! an error message.
    //!
    //! Note that the layer output type is generally not identical to the data type of the output
    //! tensor, as TensorRT may insert implicit reformatting operations to convert the former to the
    //! latter. Calling layer->setOutputType(i, type) has no effect on the data type of the i-th
    //! output tensor of layer, and users need to call layer->getOutput(i)->setType(type) to change
    //! the tensor data type. This is particularly relevant if the tensor is marked as a network
    //! output, since only setType() [but not setOutputType()] will affect the data representation
    //! in the corresponding output binding.
    //!
    //! Strongly-typed networks reject calls to method setOutputType. Instead, the output type can
    //! be set only for layers that define method setToType(). Those layers are:
    //!
    //! * ICastLayer
    //! * IDequantizeLayer
    //! * IFillLayer
    //! * IQuantizeLayer
    //!
    //! \param index the index of the output to set
    //! \param dataType the type of the output
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()
    //!
    void setOutputType(int32_t index, DataType dataType) noexcept
    {
        pass("Not Implemented", true);
        // mLayer->setOutputType(index, dataType);
    }

    //!
    //! \brief get the output type of this layer
    //!
    //! \param index the index of the output
    //!
    //! \return the output precision. If no precision has been set, DataType::kFLOAT will be
    //! returned,
    //!         unless the output type is inherently DataType::kINT32.
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()
    //!
    DataType getOutputType(int32_t index) const noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->getOutputType(index);
    }

    //!
    //! \brief whether the output type has been set for this layer
    //!
    //! \param index the index of the output
    //!
    //! \return whether the output type has been explicitly set
    //!
    //! \see setOutputType() getOutputType() resetOutputType()
    //!
    bool outputTypeIsSet(int32_t index) const noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->outputTypeIsSet(index);
    }

    //!
    //! \brief reset the output type for this layer
    //!
    //! \param index the index of the output
    //!
    //! \see setOutputType() getOutputType() outputTypeIsSet()
    //!
    void resetOutputType(int32_t index) noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->resetOutputType(index);
    }

    //!
    //! \brief Set the metadata for this layer.
    //!
    //! The metadata is emitted in the JSON returned by IEngineInspector with
    //! ProfilingVerbosity set to kDETAILED.
    //!
    //! \param metadata The per-layer metadata.
    //!
    //! \warning The string name must be null-terminated and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see getMetadata()
    //! \see getLayerInformation()
    //!
    void setMetadata(char const* metadata) noexcept
    {
        pass("Not Implemented", true);
        // mLayer->setMetadata(metadata);
    }

    //!
    //! \brief Get the metadata of the layer.
    //!
    //! \return The metadata as a null-terminated C-style string. If setMetadata() has not been
    //! called,
    //!         an empty string "" will be returned as a default value.
    //!
    //! \see setMetadata()
    //!
    char const* getMetadata() const noexcept
    {
        pass("Not Implemented", true);
        // return mLayer->getMetadata();
    }

    virtual ~ILayer() noexcept = default;

    protected:
    // apiv::VLayer* mLayer;
    LayerType type_;
    std::shared_ptr<migraphx::program> program_;
    std::vector<ITensor*> inputs_;
    std::vector<std::unique_ptr<ITensor>> outputs_;
    migraphx::instruction_ref first_ins_;
};

//!
//! \enum UnaryOperation
//!
//! \brief Enumerates the unary operations that may be performed by a Unary layer.
//!
//! Operations kNOT must have inputs of DataType::kBOOL.
//!
//! Operation kSIGN and kABS must have inputs of floating-point type, DataType::kINT8,
//! DataType::kINT32 or DataType::kINT64.
//!
//! Operation kISINF must have inputs of floating-point type.
//!
//! All other operations must have inputs of floating-point type.
//!
//! \see IUnaryLayer
//!
enum class UnaryOperation : int32_t
{
    kEXP   = 0,  //!< Exponentiation.
    kLOG   = 1,  //!< Log (base e).
    kSQRT  = 2,  //!< Square root.
    kRECIP = 3,  //!< Reciprocal.
    kABS   = 4,  //!< Absolute value.
    kNEG   = 5,  //!< Negation.
    kSIN   = 6,  //!< Sine.
    kCOS   = 7,  //!< Cosine.
    kTAN   = 8,  //!< Tangent.
    kSINH  = 9,  //!< Hyperbolic sine.
    kCOSH  = 10, //!< Hyperbolic cosine.
    kASIN  = 11, //!< Inverse sine.
    kACOS  = 12, //!< Inverse cosine.
    kATAN  = 13, //!< Inverse tangent.
    kASINH = 14, //!< Inverse hyperbolic sine.
    kACOSH = 15, //!< Inverse hyperbolic cosine.
    kATANH = 16, //!< Inverse hyperbolic tangent.
    kCEIL  = 17, //!< Ceiling.
    kFLOOR = 18, //!< Floor.
    kERF   = 19, //!< Gauss error function.
    kNOT   = 20, //!< Logical NOT.
    kSIGN = 21, //!< Sign, If input > 0, output 1; if input < 0, output -1; if input == 0, output 0.
    kROUND = 22, //!< Round to nearest even for floating-point data type.
    kISINF = 23, //!< Return true if input value equals +/- infinity for floating-point data type.
};

inline std::string trtUnaryOperationToMGXOp(const UnaryOperation op)
{
    switch(op)
    {
    case UnaryOperation::kEXP: return "exp";
    case UnaryOperation::kLOG: return "log";
    case UnaryOperation::kSQRT: return "sqrt";
    case UnaryOperation::kRECIP: return "recip";
    case UnaryOperation::kABS: return "abs";
    case UnaryOperation::kNEG: return "neg";
    case UnaryOperation::kSIN: return "sin";
    case UnaryOperation::kCOS: return "cos";
    case UnaryOperation::kTAN: return "tan";
    case UnaryOperation::kSINH: return "sinh";
    case UnaryOperation::kCOSH: return "cosh";
    case UnaryOperation::kASIN: return "asin";
    case UnaryOperation::kACOS: return "acos";
    case UnaryOperation::kATAN: return "atan";
    case UnaryOperation::kASINH: return "asinh";
    case UnaryOperation::kACOSH: return "acosh";
    case UnaryOperation::kATANH: return "atanh";
    case UnaryOperation::kCEIL: return "ceil";
    case UnaryOperation::kFLOOR: return "floor";
    case UnaryOperation::kERF: return "erf";
    case UnaryOperation::kNOT: return "unary_not";
    case UnaryOperation::kSIGN: return "sign";
    case UnaryOperation::kROUND: return "nearbyint";
    // Not included in operators.hpp
    case UnaryOperation::kISINF: return "isinf";
    }
}

//!
//! \class IUnaryLayer
//!
//! \brief Layer that represents an unary operation.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IUnaryLayer : public ILayer
{
    public:
    using ILayer::ILayer;

    IUnaryLayer(ITensor& input,
                UnaryOperation operation,
                const std::shared_ptr<migraphx::program>& program)
        : ILayer{LayerType::kUNARY, program}, operation_{operation}
    {
        auto* mm             = program->get_main_module();
        std::string unary_op = trtUnaryOperationToMGXOp(operation);
        first_ins_ = mm->add_instruction(migraphx::make_op(unary_op), input.getInstruction());
        inputs_.push_back(&input);
        outputs_.emplace_back(std::make_unique<ITensor>(first_ins_));
    }

    //!
    //! \brief Set the unary operation for the layer.
    //!
    //! When running this layer on DLA, only UnaryOperation::kABS is supported.
    //!
    //! \see getOperation(), UnaryOperation
    //!
    void setOperation(UnaryOperation operation) noexcept
    {
        auto* mm = program_->get_main_module();
        auto op  = trtUnaryOperationToMGXOp(operation);
        mm->replace_instruction(first_ins_, migraphx::make_op(op), first_ins_->inputs());
        // mImpl->setOperation(op);
    }

    //!
    //! \brief Get the unary operation for the layer.
    //!
    //! \see setOperation(), UnaryOperation
    //!
    UnaryOperation getOperation() const noexcept
    {
        return operation_;
        // return mImpl->getOperation();
    }

    virtual ~IUnaryLayer() noexcept = default;

    protected:
    UnaryOperation operation_;
    // apiv::VUnaryLayer* mImpl;
};

//!
//! \enum ElementWiseOperation
//!
//! \brief Enumerates the binary operations that may be performed by an ElementWise layer.
//!
//! Operations kAND, kOR, and kXOR must have inputs of DataType::kBOOL.
//!
//! Operation kPOW must have inputs of floating-point type or DataType::kINT8.
//!
//! All other operations must have inputs of floating-point type, DataType::kINT8, DataType::kINT32,
//! or DataType::kINT64.
//!
//! \see IElementWiseLayer
//!
enum class ElementWiseOperation : int32_t
{
    kSUM       = 0,  //!< Sum of the two elements.
    kPROD      = 1,  //!< Product of the two elements.
    kMAX       = 2,  //!< Maximum of the two elements.
    kMIN       = 3,  //!< Minimum of the two elements.
    kSUB       = 4,  //!< Subtract the second element from the first.
    kDIV       = 5,  //!< Divide the first element by the second.
    kPOW       = 6,  //!< The first element to the power of the second element.
    kFLOOR_DIV = 7,  //!< Floor division of the first element by the second.
    kAND       = 8,  //!< Logical AND of two elements.
    kOR        = 9,  //!< Logical OR of two elements.
    kXOR       = 10, //!< Logical XOR of two elements.
    kEQUAL     = 11, //!< Check if two elements are equal.
    kGREATER   = 12, //!< Check if element in first tensor is greater than corresponding element in
                     //!< second tensor.
    kLESS = 13 //!< Check if element in first tensor is less than corresponding element in second
               //!< tensor.
};

//!
//! \class IElementWiseLayer
//!
//! \brief A elementwise layer in a network definition.
//!
//! This layer applies a per-element binary operation between corresponding elements of two tensors.
//!
//! The input tensors must have the same rank. For each dimension, their lengths must
//! match, or one of them must be one. In the latter case, the tensor is broadcast along that axis.
//!
//! The output tensor has the same rank as the inputs. For each output dimension,
//! its length is equal to the lengths of the corresponding input dimensions if they match,
//! otherwise it is equal to the length that is not one.
//!
//! \warning When running this layer on the DLA with Int8 data type, the dynamic ranges of two input
//! tensors shall be equal. If the dynamic ranges are generated using calibrator, the largest value
//! shall be used.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IElementWiseLayer : public ILayer
{
    public:
    using ILayer::ILayer;

    //!
    //! \brief Set the binary operation for the layer.
    //!
    //! DLA supports only kSUM, kPROD, kMAX, kMIN, and kSUB.
    //!
    //! \see getOperation(), ElementWiseOperation
    //!
    //! \see getBiasWeights()
    //!
    void setOperation(ElementWiseOperation op) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setOperation(op);
    }

    //!
    //! \brief Get the binary operation for the layer.
    //!
    //! \see setOperation(), ElementWiseOperation
    //!
    //! \see setBiasWeights()
    //!
    ElementWiseOperation getOperation() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getOperation();
    }

    virtual ~IElementWiseLayer() noexcept = default;

    protected:
    // apiv::VElementWiseLayer* mImpl;
};

//!
//! \enum ReduceOperation
//!
//! \brief Enumerates the reduce operations that may be performed by a Reduce layer.
//!
//! The table shows the result of reducing across an empty volume of a given type.
//!
//! Operation | kFLOAT and kHALF  | kINT32  | kINT8
//! --------- | ----------------- | ------- | -----
//! kSUM      | 0                 | 0       | 0
//! kPROD     | 1                 | 1       | 1
//! kMAX      | negative infinity | INT_MIN | -128
//! kMIN      | positive infinity | INT_MAX | 127
//! kAVG      | NaN               | 0       | -128
//!
//! The current version of TensorRT usually performs reduction for kINT8 via kFLOAT or kHALF.
//! The kINT8 values show the quantized representations of the floating-point values.
//!
enum class ReduceOperation : int32_t
{
    kSUM  = 0,
    kPROD = 1,
    kMAX  = 2,
    kMIN  = 3,
    kAVG  = 4
};

//!
//! \class IReduceLayer
//!
//! \brief Layer that represents a reduction across a non-bool tensor.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IReduceLayer : public ILayer
{
    public:
    using ILayer::ILayer;

    IReduceLayer(ReduceOperation operation, const std::shared_ptr<migraphx::program>& program)
        : ILayer{LayerType::kREDUCE, program}, operation_{operation}
    {
    }

    //!
    //! \brief Set the reduce operation for the layer.
    //!
    //! \see getOperation(), ReduceOperation
    //!
    void setOperation(ReduceOperation op) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setOperation(op);
    }

    //!
    //! \brief Get the reduce operation for the layer.
    //!
    //! \see setOperation(), ReduceOperation
    //!
    ReduceOperation getOperation() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getOperation();
    }

    //!
    //! \brief Set the axes over which to reduce.
    //!
    //! \see getReduceAxes
    //!
    void setReduceAxes(uint32_t reduceAxes) noexcept
    {
        auto ins = inputs_.front()->getInstruction();
        ins->replace(migraphx::make_op(ins->name(), {{"axes", axesToVector(reduceAxes)}}));
        // mImpl->setReduceAxes(reduceAxes);
    }

    //!
    //! \brief Get the axes over which to reduce for the layer.
    //!
    //! \see setReduceAxes
    //!
    uint32_t getReduceAxes() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getReduceAxes();
    }

    //!
    //! \brief Set the boolean that specifies whether or not to keep the reduced dimensions for the
    //! layer.
    //!
    //! \see getKeepDimensions
    //!
    void setKeepDimensions(bool keepDimensions) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setKeepDimensions(keepDimensions);
    }

    //!
    //! \brief Get the boolean that specifies whether or not to keep the reduced dimensions for the
    //! layer.
    //!
    //! \see setKeepDimensions
    //!
    bool getKeepDimensions() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getKeepDimensions();
    }

    virtual ~IReduceLayer() noexcept = default;

    protected:
    ReduceOperation operation_;
    // apiv::VReduceLayer* mImpl;
};

//!
//! \class IHostMemory
//!
//! \brief Class to handle library allocated memory that is accessible to the user.
//!
//! The memory allocated via the host memory object is owned by the library and will
//! be de-allocated when the destroy method is called.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IHostMemory : public INoCopy
{
    public:
    virtual ~IHostMemory() noexcept = default;

    //! A pointer to the raw data that is owned by the library.
    void* data() const noexcept { return data_; }

    //! The size in bytes of the data that was allocated.
    std::size_t size() const noexcept { return size_; }

    //! The type of the memory that was allocated.
    DataType type() const noexcept { return type_; }

    IHostMemory(void* data, size_t size, DataType type) : data_{data}, size_{size}, type_{type} {}

    protected:
    // apiv::VHostMemory* mImpl;
    void* data_;
    size_t size_;
    DataType type_;
};

//!
//! \enum TempfileControlFlag
//!
//! \brief Flags used to control TensorRT's behavior when creating executable temporary files.
//!
//! On some platforms the TensorRT runtime may need to create files in a temporary directory or use
//! platform-specific APIs to create files in-memory to load temporary DLLs that implement runtime
//! code. These flags allow the application to explicitly control TensorRT's use of these files.
//! This will preclude the use of certain TensorRT APIs for deserializing and loading lean runtimes.
//!
enum class TempfileControlFlag : int32_t
{
    //! Allow creating and loading files in-memory (or unnamed files).
    kALLOW_IN_MEMORY_FILES = 0,

    //! Allow creating and loading named files in a temporary directory on the filesystem.
    //!
    //! \see IRuntime::setTemporaryDirectory()
    kALLOW_TEMPORARY_FILES = 1,
};

//! Maximum number of elements in TempfileControlFlag enum. \see TempfileControlFlag
template <>
constexpr inline int32_t EnumMax<TempfileControlFlag>() noexcept
{
    return 2;
}

//!
//! \brief Represents a collection of one or more TempfileControlFlag values combined using
//! bitwise-OR operations.
//!
//! \see TempfileControlFlag,
//!      IRuntime::setTempfileControlFlags(),
//!      IRuntime::getTempfileControlFlags()
using TempfileControlFlags = uint32_t;

//!
//! \enum ExecutionContextAllocationStrategy
//!
//! \brief Different memory allocation behaviors for IExecutionContext.
//!
//! IExecutionContext requires a block of device memory for internal activation tensors during
//! inference. The user can either let the execution context manage the memory in various ways or
//! allocate the memory themselves.
//!
//! \see ICudaEngine::createExecutionContext()
//! \see IExecutionContext::setDeviceMemory()
//!
enum class ExecutionContextAllocationStrategy : int32_t
{
    kSTATIC = 0, //!< Default static allocation with the maximum size across all profiles.
    kON_PROFILE_CHANGE = 1, //!< Reallocate for a profile when it's selected.
    kUSER_MANAGED      = 2, //!< The user supplies custom allocation to the execution context.
};

//!
//! \enum OptProfileSelector
//!
//! \brief When setting or querying optimization profile parameters (such as shape tensor inputs or
//! dynamic dimensions),
//!        select whether we are interested in the minimum, optimum, or maximum values for these
//!        parameters. The minimum and maximum specify the permitted range that is supported at
//!        runtime, while the optimum value is used for the kernel selection. This should be the
//!        "typical" value that is expected to occur at runtime.
//!
//! \see IOptimizationProfile::setDimensions(), IOptimizationProfile::setShapeValues()
//!
enum class OptProfileSelector : int32_t
{
    kMIN =
        0, //!< This is used to set or get the minimum permitted value for dynamic dimensions etc.
    kOPT = 1, //!< This is used to set or get the value that is used in the optimization (kernel
              //!< selection).
    kMAX = 2 //!< This is used to set or get the maximum permitted value for dynamic dimensions etc.
};

//!
//! \brief Number of different values of OptProfileSelector enum.
//!
//! \see OptProfileSelector
//!
template <>
constexpr inline int32_t EnumMax<OptProfileSelector>() noexcept
{
    return 3;
}

//!
//! \enum EngineCapability
//!
//! \brief List of supported engine capability flows.
//!
//! \details The EngineCapability determines the restrictions of a network during build time and
//! what runtime it targets. When BuilderFlag::kSAFETY_SCOPE is not set (by default),
//! EngineCapability::kSTANDARD does not provide any restrictions on functionality and the resulting
//! serialized engine can be executed with TensorRT's standard runtime APIs in the nvinfer1
//! namespace. EngineCapability::kSAFETY provides a restricted subset of network operations that are
//! safety certified and the resulting serialized engine can be executed with TensorRT's safe
//! runtime APIs in the nvinfer1::safe namespace. EngineCapability::kDLA_STANDALONE provides a
//! restricted subset of network operations that are DLA compatible and the resulting serialized
//! engine can be executed using standalone DLA runtime APIs. See sampleCudla for an example of
//! integrating cuDLA APIs with TensorRT APIs.
//!
enum class EngineCapability : int32_t
{
    //!
    //! Standard: TensorRT flow without targeting the safety runtime.
    //! This flow supports both DeviceType::kGPU and DeviceType::kDLA.
    //!
    kSTANDARD = 0,

    //!
    //! Safety: TensorRT flow with restrictions targeting the safety runtime.
    //! See safety documentation for list of supported layers and formats.
    //! This flow supports only DeviceType::kGPU.
    //!
    //! This flag is only supported in NVIDIA Drive(R) products.
    kSAFETY = 1,

    //!
    //! DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA
    //! runtimes. See DLA documentation for list of supported layers and formats. This flow supports
    //! only DeviceType::kDLA.
    //!
    kDLA_STANDALONE = 2,
};

//!
//! \enum TacticSource
//!
//! \brief List of tactic sources for TensorRT.
//!
//! \see TacticSources, IBuilderConfig::setTacticSources(), IBuilderConfig::getTacticSources()
//!
enum class TacticSource : int32_t
{
    //! cuBLAS tactics. Disabled by default.
    //! \note Disabling kCUBLAS will cause the cuBLAS handle passed to plugins in attachToContext to
    //! be null. \deprecated Deprecated in TensorRT 10.0.
    kCUBLAS = 0,

    //! cuBLAS LT tactics. Enabled by default.
    //! \deprecated Deprecated in TensorRT 9.0.
    kCUBLAS_LT = 1,

    //! cuDNN tactics. Disabled by default.
    //! \note Disabling kCUDNN will cause the cuDNN handle passed to plugins in attachToContext to
    //! be null. \deprecated Deprecated in TensorRT 10.0.
    kCUDNN = 2,

    //! Enables convolution tactics implemented with edge mask tables. These tactics tradeoff memory
    //! for performance by consuming additional memory space proportional to the input size. Enabled
    //! by default.
    kEDGE_MASK_CONVOLUTIONS = 3,

    //! Enables convolution tactics implemented with source-code JIT fusion. The engine building
    //! time may increase when this is enabled. Enabled by default.
    kJIT_CONVOLUTIONS = 4,
};

//!
//! \brief Represents a collection of one or more TacticSource values
//! combine using bitwise-OR operations.
//!
//! \see IBuilderConfig::setTacticSources(), IBuilderConfig::getTacticSources()
//!
using TacticSources = uint32_t;

//!
//! \enum ProfilingVerbosity
//!
//! \brief List of verbosity levels of layer information exposed in NVTX annotations and in
//! IEngineInspector.
//!
//! \see IBuilderConfig::setProfilingVerbosity(),
//!      IBuilderConfig::getProfilingVerbosity(),
//!      IEngineInspector
//!
enum class ProfilingVerbosity : int32_t
{
    kLAYER_NAMES_ONLY = 0, //!< Print only the layer names. This is the default setting.
    kNONE             = 1, //!< Do not print any layer information.
    kDETAILED = 2, //!< Print detailed layer information including layer names and layer parameters.
};

//!
//! \enum HardwareCompatibilityLevel
//!
//! \brief Describes requirements of compatibility with GPU architectures other than that of the GPU
//! on which the engine was built.
//!
//! Levels except kNONE are only supported for engines built on NVIDIA Ampere and later GPUs.
//!
//! \warning Note that compatibility with future hardware depends on CUDA forward compatibility
//! support.
//!
enum class HardwareCompatibilityLevel : int32_t
{
    //! Do not require hardware compatibility with GPU architectures other than that of the GPU on
    //! which the engine was built.
    kNONE = 0,

    //! Require that the engine is compatible with Ampere and newer GPUs. This will limit the
    //! combined usage of driver reserved and backend kernel max shared memory to 48KiB, may reduce
    //! the number of available tactics for each layer, and may prevent some fusions from occurring.
    //! Thus this can decrease the performance, especially for tf32 models. This option will disable
    //! cuDNN, cuBLAS, and cuBLAS LT as tactic sources.
    //!
    //! The driver reserved shared memory can be queried from cuDeviceGetAttribute(&reservedShmem,
    //! CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK).
    //!
    kAMPERE_PLUS = 1,
};

//!
//! \class IExecutionContext
//!
//! \brief Context for executing inference using an engine, with functionally unsafe features.
//!
//! Multiple execution contexts may exist for one ICudaEngine instance, allowing the same
//! engine to be used for the execution of multiple batches simultaneously. If the engine supports
//! dynamic shapes, each execution context in concurrent use must use a separate optimization
//! profile.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
class IExecutionContext : public INoCopy
{
    public:
    virtual ~IExecutionContext() noexcept = default;

    //!
    //! \brief Set the debug sync flag.
    //!
    //! If this flag is set to true, the engine will log the successful execution for each kernel
    //! during executeV2(). It has no effect when using enqueueV3().
    //!
    //! \see getDebugSync()
    //!
    void setDebugSync(bool sync) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDebugSync(sync);
    }

    //!
    //! \brief Get the debug sync flag.
    //!
    //! \see setDebugSync()
    //!
    bool getDebugSync() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDebugSync();
    }

    //!
    //! \brief Set the profiler.
    //!
    //! \see IProfiler getProfiler()
    //!
    void setProfiler(IProfiler* profiler) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setProfiler(profiler);
    }

    //!
    //! \brief Get the profiler.
    //!
    //! \see IProfiler setProfiler()
    //!
    IProfiler* getProfiler() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProfiler();
    }

    //!
    //! \brief Get the associated engine.
    //!
    //! \see ICudaEngine
    //!
    ICudaEngine const& getEngine() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getEngine();
    }

    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see getName()
    //!
    void setName(char const* name) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setName(name);
    }

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \see setName()
    //!
    char const* getName() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getName();
    }

    //!
    //! \brief Set the device memory for use by this execution context.
    //!
    //! The memory must be aligned with cuda memory alignment property (using
    //! cudaGetDeviceProperties()), and its size must be large enough for performing inference with
    //! the given network inputs. getDeviceMemorySize() and getDeviceMemorySizeForProfile() report
    //! upper bounds of the size. Setting memory to nullptr is acceptable if the reported size is 0.
    //! If using enqueueV3() to run the network, the memory is in use from the invocation of
    //! enqueueV3() until network execution is complete. If using executeV2(), it is in use until
    //! executeV2() returns. Releasing or otherwise using the memory for other purposes during this
    //! time will result in undefined behavior.
    //!
    //! \see ICudaEngine::getDeviceMemorySize()
    //! \see ICudaEngine::getDeviceMemorySizeForProfile()
    //! \see ExecutionContextAllocationStrategy
    //! \see ICudaEngine::createExecutionContext()
    //! \see ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    void setDeviceMemory(void* memory) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDeviceMemory(memory);
    }

    //!
    //! \brief Return the strides of the buffer for the given tensor name.
    //!
    //! The strides are in units of elements, not components or bytes.
    //! For example, for TensorFormat::kHWC8, a stride of one spans 8 scalars.
    //!
    //! Note that strides can be different for different execution contexts
    //! with dynamic shapes.
    //!
    //! If the provided name does not map to an input or output tensor, or there are dynamic
    //! dimensions that have not been set yet, return Dims{-1, {}}
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    Dims getTensorStrides(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorStrides(tensorName);
    }

    public:
    //!
    //! \brief Get the index of the currently selected optimization profile.
    //!
    //! If the profile index has not been set yet (implicitly to 0 if no other execution context has
    //! been set to profile 0, or explicitly for all subsequent contexts), an invalid value of -1
    //! will be returned and all calls to enqueueV3()/executeV2() will fail until a valid profile
    //! index has been set. This behavior is deprecated in TensorRT 8.6, all profiles will default
    //! to optimization profile 0 and -1 will no longer be returned.
    //!
    int32_t getOptimizationProfile() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getOptimizationProfile();
    }

    //!
    //! \brief Set shape of given input.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param dims The shape of an input tensor.
    //!
    //! \return True on success, false if the provided name does not map to an input tensor, or if
    //! some other error occurred.
    //!
    //! Each dimension must agree with the network dimension unless the latter was -1.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    bool setInputShape(char const* tensorName, Dims const& dims) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setInputShape(tensorName, dims);
    }

    //!
    //! \brief Return the shape of the given input or output.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! Return Dims{-1, {}} if the provided name does not map to an input or output tensor.
    //! Otherwise return the shape of the input or output tensor.
    //!
    //! A dimension in an input tensor will have a -1 wildcard value if all the following are true:
    //!  * setInputShape() has not yet been called for this tensor
    //!  * The dimension is a runtime dimension that is not implicitly constrained to be a single
    //!  value.
    //!
    //! A dimension in an output tensor will have a -1 wildcard value if the dimension depends
    //! on values of execution tensors OR if all the following are true:
    //!  * It is a runtime dimension.
    //!  * setInputShape() has NOT been called for some input tensor(s) with a runtime shape.
    //!  * setTensorAddress() has NOT been called for some input tensor(s) with isShapeInferenceIO()
    //!  = true.
    //!
    //! An output tensor may also have -1 wildcard dimensions if its shape depends on values of
    //! tensors supplied to enqueueV3().
    //!
    //! If the request is for the shape of an output tensor with runtime dimensions,
    //! all input tensors with isShapeInferenceIO() = true should have their value already set,
    //! since these values might be needed to compute the output shape.
    //!
    //! Examples of an input dimension that is implicitly constrained to a single value:
    //! * The optimization profile specifies equal min and max values.
    //! * The dimension is named and only one value meets the optimization profile requirements
    //!   for dimensions with that name.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    Dims getTensorShape(char const* tensorName) const noexcept
    {
        return toDimensions(program_->get_parameter_shapes().at(tensorName));
        // return mImpl->getTensorShape(tensorName);
    }

    //!
    //! \brief Whether all dynamic dimensions of input tensors have been specified
    //!
    //! \return True if all dynamic dimensions of input tensors have been specified
    //!         by calling setInputShape().
    //!
    //! Trivially true if network has no dynamically shaped input tensors.
    //!
    //! Does not work with name-base interfaces eg. IExecutionContext::setInputShape(). Use
    //! IExecutionContext::inferShapes() instead.
    //!
    bool allInputDimensionsSpecified() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->allInputDimensionsSpecified();
    }

    //!
    //! \brief Whether all input shape bindings have been specified
    //!
    //! \return True if all input shape bindings have been specified by setInputShapeBinding().
    //!
    //! Trivially true if network has no input shape bindings.
    //!
    //! Does not work with name-base interfaces eg. IExecutionContext::setInputShape(). Use
    //! IExecutionContext::inferShapes() instead.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. setInputShapeBinding() is removed since
    //! TensorRT 10.0.
    //!
    [[deprecated]] bool allInputShapesSpecified() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->allInputShapesSpecified();
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during
    //! execution. This function will call incRefCount of the registered ErrorRecorder at least
    //! once. Setting recorder to nullptr unregisters the recorder with the interface, resulting in
    //! a call to decRefCount if a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //!
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned
    //! if an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Synchronously execute a network.
    //!
    //! This method requires an array of input and output buffers. The mapping
    //! from indices to tensor names can be queried using ICudaEngine::getIOTensorName().
    //!
    //! \param bindings An array of pointers to input and output buffers for the network.
    //!
    //! \return True if execution succeeded.
    //!
    //! \see ICudaEngine::getIOTensorName()
    //!
    bool executeV2(void* const* bindings) noexcept
    {
        // incorrect implementation for V2
        auto result = program_->eval(param_map_);
        return true;
        // return mImpl->executeV2(bindings);
    }

    //!
    //! \brief Select an optimization profile for the current context with async
    //! semantics.
    //!
    //! \param profileIndex Index of the profile. The value must lie between 0 and
    //!        getEngine().getNbOptimizationProfiles() - 1
    //!
    //! \param stream A cuda stream on which the cudaMemcpyAsyncs may be
    //! enqueued
    //!
    //! When an optimization profile is switched via this API, TensorRT may
    //! require that data is copied via cudaMemcpyAsync. It is the
    //! application’s responsibility to guarantee that synchronization between
    //! the profile sync stream and the enqueue stream occurs.
    //!
    //! The selected profile will be used in subsequent calls to executeV2()/enqueueV3().
    //! If the associated CUDA engine has inputs with dynamic shapes, the optimization profile must
    //! be set with its corresponding profileIndex before calling execute or enqueue. The newly
    //! created execution context will be assigned optimization profile 0.
    //!
    //! If the associated CUDA engine does not have inputs with dynamic shapes,
    //! this method need not be called, in which case the default profile index
    //! of 0 will be used.
    //!
    //! setOptimizationProfileAsync() must be called before calling
    //! setInputShape() for all dynamic input
    //! tensors or input shape tensors, which in turn must be called before
    //! executeV2()/enqueueV3().
    //!
    //! \warning This function will trigger layer resource updates on the next call of
    //!          executeV2()/enqueueV3(), possibly resulting in performance bottlenecks.
    //!
    //! \warning Not synchronizing the stream used at enqueue with the stream
    //! used to set optimization profile asynchronously using this API will
    //! result in undefined behavior.
    //!
    //! \return true if the call succeeded, else false (e.g. input out of range)
    //!
    //! \see ICudaEngine::getNbOptimizationProfiles()
    bool setOptimizationProfileAsync(int32_t profileIndex, hipStream_t stream) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setOptimizationProfileAsync(profileIndex, stream);
    }

    //!
    //! \brief Set whether enqueue emits layer timing to the profiler
    //!
    //! If set to true (default), enqueue is synchronous and does layer timing profiling implicitly
    //! if there is a profiler attached. If set to false, enqueue will be asynchronous if there is a
    //! profiler attached. An extra method reportToProfiler() needs to be called to obtain the
    //! profiling data and report to the profiler attached.
    //!
    //! \see IExecutionContext::getEnqueueEmitsProfile()
    //! \see IExecutionContext::reportToProfiler()
    //!
    void setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setEnqueueEmitsProfile(enqueueEmitsProfile);
    }

    //!
    //! \brief Get the enqueueEmitsProfile state.
    //!
    //! \return The enqueueEmitsProfile state.
    //!
    //! \see IExecutionContext::setEnqueueEmitsProfile()
    //!
    bool getEnqueueEmitsProfile() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getEnqueueEmitsProfile();
    }

    //!
    //! \brief Calculate layer timing info for the current optimization profile in IExecutionContext
    //! and update the profiler after one iteration of inference launch.
    //!
    //! If IExecutionContext::getEnqueueEmitsProfile() returns true, the enqueue function will
    //! calculate layer timing implicitly if a profiler is provided. This function returns true and
    //! does nothing.
    //!
    //! If IExecutionContext::getEnqueueEmitsProfile() returns false, the enqueue function will
    //! record the CUDA event timers if a profiler is provided. But it will not perform the layer
    //! timing calculation. IExecutionContext::reportToProfiler() needs to be called explicitly to
    //! calculate layer timing for the previous inference launch.
    //!
    //! In the CUDA graph launch scenario, it will record the same set of CUDA events
    //! as in regular enqueue functions if the graph is captured from an IExecutionContext with
    //! profiler enabled. This function needs to be called after graph launch to report the layer
    //! timing info to the profiler.
    //!
    //! \warning profiling CUDA graphs is only available from CUDA 11.1 onwards.
    //! \warning reportToProfiler uses the stream of the previous enqueue call, so the stream must
    //! be live otherwise behavior is undefined.
    //!
    //! \return true if the call succeeded, else false (e.g. profiler not provided, in CUDA graph
    //! capture mode, etc.)
    //!
    //! \see IExecutionContext::setEnqueueEmitsProfile()
    //! \see IExecutionContext::getEnqueueEmitsProfile()
    //!
    bool reportToProfiler() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->reportToProfiler();
    }

    //!
    //! \brief Set memory address for given input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param data The pointer (void*) to the data owned by the user.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! An address defaults to nullptr.
    //! Pass data=nullptr to reset to the default state.
    //!
    //! Return false if the provided name does not map to an input or output tensor.
    //!
    //! If an input pointer has type (void const*), use setInputTensorAddress() instead.
    //!
    //! Before calling enqueueV3(), each input must have a non-null address and
    //! each output must have a non-null address or an IOutputAllocator to set it later.
    //!
    //! If the TensorLocation of the tensor is kHOST, the pointer must point to a host buffer of
    //! sufficient size. If the TensorLocation of the tensor is kDEVICE, the pointer must point to a
    //! device buffer of sufficient size and alignment, or be nullptr if the tensor is an output
    //! tensor that will be allocated by IOutputAllocator.
    //!
    //! If getTensorShape(name) reports a -1 for any dimension of an output after all
    //! input shapes have been set, then to find out
    //! the dimensions, use setOutputAllocator() to associate an IOutputAllocator to
    //! which the dimensions will be reported when known.
    //!
    //! Calling both setTensorAddress and setOutputAllocator() for the same output is allowed,
    //! and can be useful for preallocating memory, and then reallocating if it's not big enough.
    //!
    //! The pointer must have at least 256-byte alignment.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see setInputTensorAddress() setOutputTensorAddress() getTensorShape() setOutputAllocator()
    //! IOutputAllocator
    //!
    bool setTensorAddress(char const* tensorName, void* data) noexcept
    {
        // TODO
        param_map_[tensorName] =
            migraphx::argument(program_->get_parameter_shapes().at(tensorName), data);
        return true;
        // return mImpl->setTensorAddress(tensorName, data);
    }

    //!
    //! \brief Get memory address bound to given input or output tensor, or nullptr if the provided
    //! name does not map to an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! Use method getOutputTensorAddress() if a non-const pointer for an output tensor is required.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see getOutputTensorAddress()
    //!
    void const* getTensorAddress(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorAddress(tensorName);
    }

    //!
    //! \brief Set the memory address for a given output tensor.
    //!
    //! \param tensorName The name of an output tensor.
    //! \param data The pointer to the buffer to which to write the output.
    //!
    //! \return True on success, false if the provided name does not map to an output tensor, does
    //! not meet alignment requirements, or some other error occurred.
    //!
    //! Output addresses can also be set using method setTensorAddress. This method is provided for
    //! applications which prefer to use different methods for setting input and output tensors.
    //!
    //! See setTensorAddress() for alignment and data type constraints.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see setTensorAddress()
    //!
    bool setOutputTensorAddress(char const* tensorName, void* data) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setOutputTensorAddress(tensorName, data);
    }

    //!
    //! \brief Set memory address for given input.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param data The pointer (void const*) to the const data owned by the user.
    //!
    //! \return True on success, false if the provided name does not map to an input tensor, does
    //! not meet alignment requirements, or some other error occurred.
    //!
    //! Input addresses can also be set using method setTensorAddress, which requires a (void*).
    //!
    //! See description of method setTensorAddress() for alignment and data type constraints.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see setTensorAddress()
    //!
    bool setInputTensorAddress(char const* tensorName, void const* data) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setInputTensorAddress(tensorName, data);
    }

    //!
    //! \brief Get memory address for given output.
    //!
    //! \param tensorName The name of an output tensor.
    //!
    //! \return Raw output data pointer (void*) for given output tensor, or nullptr if the provided
    //! name does not map to an output tensor.
    //!
    //! If only a (void const*) pointer is needed, an alternative is to call method
    //! getTensorAddress().
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see getTensorAddress()
    //!
    void* getOutputTensorAddress(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getOutputTensorAddress(tensorName);
    }

    //!
    //! \brief Run shape calculations.
    //!
    //! \param nbMaxNames Maximum number of names to write to tensorNames.
    //!        When the return value is a positive value n and tensorNames != nullptr,
    //!        the names of min(n,nbMaxNames) insufficiently specified input tensors are
    //!        written to tensorNames.
    //!
    //! \param tensorNames Buffer in which to place names of insufficiently specified input tensors.
    //!
    //! \return 0 on success.
    //!         Positive value n if n input tensors were not sufficiently specified.
    //!         -1 for other errors.
    //!
    //! An input tensor is insufficiently specified if either of the following is true:
    //!
    //! * It has dynamic dimensions and its runtime dimensions have not yet
    //!   been specified via IExecutionContext::setInputShape.
    //!
    //! * isShapeInferenceIO(t)=true and the tensor's address has not yet been set.
    //!
    //! If an output tensor has isShapeInferenceIO(t)=true and its address has been specified,
    //! then its value is written.
    //!
    //! Returns -1 if tensorNames == nullptr and nbMaxNames != 0.
    //! Returns -1 if nbMaxNames < 0.
    //! Returns -1 if a tensor's dimensions are invalid, e.g. a tensor ends up with a negative
    //! dimension.
    //!
    int32_t inferShapes(int32_t nbMaxNames, char const** tensorNames) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->inferShapes(nbMaxNames, tensorNames);
    }

    //!
    //! \brief Recompute the internal activation buffer sizes based on the current input shapes, and
    //! return the total amount of memory required.
    //!
    //! Users can allocate the device memory based on the size returned and provided the memory to
    //! TRT with IExecutionContext::setDeviceMemory(). Must specify all input shapes and the
    //! optimization profile to use before calling this function, otherwise the partition will be
    //! invalidated.
    //!
    //! \return Total amount of memory required on success, 0 if error occurred.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    size_t updateDeviceMemorySizeForShapes() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->updateDeviceMemorySizeForShapes();
    }

    //!
    //! \brief Mark input as consumed.
    //!
    //! \param event The cuda event that is triggered after all input tensors have been consumed.
    //!
    //! \warning The set event must be valid during the inferece.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! Passing event==nullptr removes whatever event was set, if any.
    //!
    bool setInputConsumedEvent(hipEvent_t event) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setInputConsumedEvent(event);
    }

    //!
    //! \brief The event associated with consuming the input.
    //!
    //! \return The cuda event. Nullptr will be returned if the event is not set yet.
    //!
    hipEvent_t getInputConsumedEvent() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getInputConsumedEvent();
    }

    //!
    //! \brief Set output allocator to use for output tensor of given name.
    //! Pass nullptr to outputAllocator to unset.
    //! The allocator is called by enqueueV3().
    //!
    //! \param tensorName The name of an output tensor.
    //! \param outputAllocator IOutputAllocator for the tensors.
    //!
    //! \return True if success, false if the provided name does not map to an output or, if some
    //! other error occurred.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see enqueueV3() IOutputAllocator
    //!
    bool setOutputAllocator(char const* tensorName, IOutputAllocator* outputAllocator) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setOutputAllocator(tensorName, outputAllocator);
    }

    //!
    //! \brief Get output allocator associated with output tensor of given name, or nullptr if the
    //! provided name does not map to an output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see IOutputAllocator
    //!
    IOutputAllocator* getOutputAllocator(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getOutputAllocator(tensorName);
    }

    //!
    //! \brief Get upper bound on an output tensor's size, in bytes, based on
    //! the current optimization profile and input dimensions.
    //!
    //! If the profile or input dimensions are not yet set, or the provided name
    //! does not map to an output, returns -1.
    //!
    //! \param tensorName The name of an output tensor.
    //!
    //! \return Upper bound in bytes.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    int64_t getMaxOutputSize(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMaxOutputSize(tensorName);
    }

    //!
    //! \brief Specify allocator to use for internal temporary storage.
    //!
    //! This allocator is used only by enqueueV3() for temporary storage whose size cannot be
    //! predicted ahead of enqueueV3(). It is not used for output tensors, because memory
    //! allocation for those is allocated by the allocator set by setOutputAllocator().
    //! All memory allocated is freed by the time enqueueV3() returns.
    //!
    //! \param allocator pointer to allocator to use. Pass nullptr to revert to using TensorRT's
    //!        default allocator.
    //!
    //! \return True on success, false if error occurred.
    //!
    //! \see enqueueV3() setOutputAllocator()
    //!
    bool setTemporaryStorageAllocator(IGpuAllocator* allocator) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setTemporaryStorageAllocator(allocator);
    }

    //!
    //! \brief Get allocator set by setTemporaryStorageAllocator.
    //!
    //! Returns a nullptr if a nullptr was passed with setTemporaryStorageAllocator().
    //!
    IGpuAllocator* getTemporaryStorageAllocator() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTemporaryStorageAllocator();
    }

    //!
    //! \brief Enqueue inference on a stream.
    //!
    //! \param stream A cuda stream on which the inference kernels will be enqueued.
    //!
    //! \return True if the kernels were enqueued successfully, false otherwise.
    //!
    //! Modifying or releasing memory that has been registered for the tensors before stream
    //! synchronization or the event passed to setInputConsumedEvent has been being triggered
    //! results in undefined behavior. Input tensor can be released after the setInputConsumedEvent
    //! whereas output tensors require stream synchronization.
    //!
    //! \warning Using default stream may lead to performance issues due to additional
    //! cudaDeviceSynchronize() calls by
    //!          TensorRT to ensure correct synchronizations. Please use non-default stream instead.
    //!
    //! \warning If the Engine is streaming weights, enqueueV3 will become synchronous, and
    //!          the graph will not be capturable.
    //!
    bool enqueueV3(hipStream_t stream) noexcept
    {
        migraphx::execution_environment exec_env{
            migraphx::any_ptr(reinterpret_cast<void*>(stream), "ihipStream_t"), true};
        auto result = program_->eval(param_map_, exec_env);
        return true;
        // return mImpl->enqueueV3(stream);
    }

    //!
    //! \brief Set the maximum size for persistent cache usage.
    //!
    //! This function sets the maximum persistent L2 cache that this execution context may use for
    //! activation caching. Activation caching is not supported on all architectures - see "How
    //! TensorRT uses Memory" in the developer guide for details
    //!
    //! \param size the size of persistent cache limitation in bytes.
    //! The default is 0 Bytes.
    //!
    //! \see getPersistentCacheLimit
    void setPersistentCacheLimit(size_t size) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setPersistentCacheLimit(size);
    }

    //!
    //! \brief Get the maximum size for persistent cache usage.
    //!
    //! \returns The size of the persistent cache limit
    //!
    //! \see setPersistentCacheLimit
    size_t getPersistentCacheLimit() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getPersistentCacheLimit();
    }

    //!
    //! \brief Set the verbosity of the NVTX markers in the execution context.
    //!
    //! Building with kDETAILED verbosity will generally increase latency in enqueueV3(). Call this
    //! method to select NVTX verbosity in this execution context at runtime.
    //!
    //! The default is the verbosity with which the engine was built, and the verbosity may not be
    //! raised above that level.
    //!
    //! This function does not affect how IEngineInspector interacts with the engine.
    //!
    //! \param verbosity The verbosity of the NVTX markers.
    //!
    //! \return True if the NVTX verbosity is set successfully. False if the provided verbosity
    //! level is higher than the profiling verbosity of the corresponding engine.
    //!
    //! \see getNvtxVerbosity()
    //! \see ICudaEngine::getProfilingVerbosity()
    //!
    bool setNvtxVerbosity(ProfilingVerbosity verbosity) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setNvtxVerbosity(verbosity);
    }

    //!
    //! \brief Get the NVTX verbosity of the execution context.
    //!
    //! \return The current NVTX verbosity of the execution context.
    //!
    //! \see setNvtxVerbosity()
    //!
    ProfilingVerbosity getNvtxVerbosity() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNvtxVerbosity();
    }

    //!
    //! \brief Set the auxiliary streams that TensorRT should launch kernels on in the next
    //! enqueueV3() call.
    //!
    //! If set, TensorRT will launch the kernels that are supposed to run on the auxiliary streams
    //! using the streams provided by the user with this API. If this API is not called before the
    //! enqueueV3() call, then TensorRT will use the auxiliary streams created by TensorRT
    //! internally.
    //!
    //! TensorRT will always insert event synchronizations between the main stream provided via
    //! enqueueV3() call and the auxiliary streams:
    //!  - At the beginning of the enqueueV3() call, TensorRT will make sure that all the auxiliary
    //!  streams wait on
    //!    the activities on the main stream.
    //!  - At the end of the enqueueV3() call, TensorRT will make sure that the main stream wait on
    //!  the activities on
    //!    all the auxiliary streams.
    //!
    //! \param auxStreams The pointer to an array of cudaStream_t with the array length equal to
    //! nbStreams. \param nbStreams The number of auxiliary streams provided. If nbStreams is
    //! greater than
    //!        `engine->getNbAuxStreams()`, then only the first `engine->getNbAuxStreams()` streams
    //!        will be used. If `nbStreams` is less than `engine->getNbAuxStreams()`, such as
    //!        setting `nbStreams` to 0, then TensorRT will use the provided streams for the first
    //!        `nbStreams` auxiliary streams, and will create additional streams internally for the
    //!        rest of the auxiliary streams.
    //!
    //! \note The provided auxiliary streams must not be the default stream and must all be
    //! different to avoid
    //!       deadlocks.
    //!
    //! \see enqueueV3(), IBuilderConfig::setMaxAuxStreams(), ICudaEngine::getNbAuxStreams()
    //!
    void setAuxStreams(hipStream_t* auxStreams, int32_t nbStreams) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setAuxStreams(auxStreams, nbStreams);
    }

    //!
    //! \brief Set DebugListener for this execution context.
    //!
    //! \param listener DebugListener for this execution context.
    //!
    //! \return true if succeed, false if failure.
    //!
    bool setDebugListener(IDebugListener* listener) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setDebugListener(listener);
    }

    //!
    //! \brief Get the DebugListener of this execution context.
    //!
    //! \return DebugListener of this execution context.
    //!
    IDebugListener* getDebugListener() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDebugListener();
    }

    //!
    //! \brief Set debug state of tensor given the tensor name.
    //!
    //! Turn the debug state of a tensor on or off.
    //! A tensor with the parameter tensor name must exist in the network, and the tensor must have
    //! been marked as a debug tensor during build time. Otherwise, an error is thrown.
    //!
    //! \param name Name of target tensor.
    //!
    //! \param flag True if turning on debug state, false if turning off debug state of tensor
    //! The default is off.
    //!
    //! \return True if successful, false otherwise.
    //!
    bool setTensorDebugState(char const* name, bool flag) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setTensorDebugState(name, flag);
    }

    //!
    //! Turn the debug state of all debug tensors on or off.
    //!
    //! \param flag true if turning on debug state, false if turning off debug state.
    //!
    //! \return true if successful, false otherwise.
    //!
    //! The default is off.
    bool setAllTensorsDebugState(bool flag) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setAllTensorsDebugState(flag);
    }

    //!
    //! Get the debug state.
    //!
    //! \return true if there is a debug tensor with the given name and it has debug state turned
    //! on.
    //!
    bool getDebugState(char const* name) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDebugState(name);
    }

    IExecutionContext(const std::shared_ptr<migraphx::program>& program) : program_{program} {}

    protected:
    std::shared_ptr<migraphx::program> program_;
    migraphx::parameter_map param_map_;
};

//!
//! \class ICudaEngine
//!
//! \brief An engine for executing inference on a built network, with functionally unsafe features.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class ICudaEngine : public INoCopy
{
    public:
    ICudaEngine(const std::shared_ptr<migraphx::program>& program)
        : program_{program}, tensor_names_{program->get_parameter_names()}
    {
    }

    virtual ~ICudaEngine() noexcept = default;

    //!
    //! \brief Get shape of an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return shape of the tensor, with -1 in place of each dynamic runtime dimension,
    //!         or Dims{-1, {}} if the provided name does not map to an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    Dims getTensorShape(char const* tensorName) const noexcept
    {
        return toDimensions(program_->get_parameter_shapes().at(tensorName));
        // return mImpl->getTensorShape(tensorName);
    }

    //!
    //! \brief Determine the required data type for a buffer from its tensor name.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return The type of the data in the buffer, or DataType::kFLOAT if the provided name does
    //! not map to an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    DataType getTensorDataType(char const* tensorName) const noexcept
    {
        return toDataType(program_->get_parameter_shapes().at(tensorName).type());
        // return mImpl->getTensorDataType(tensorName);
    }

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! The number of layers in the network is not necessarily the number in the original network
    //! definition, as layers may be combined or eliminated as the engine is optimized. This value
    //! can be useful when building per-layer tables, such as when aggregating profiling data over a
    //! number of executions.
    //!
    //! \return The number of layers in the network.
    //!
    int32_t getNbLayers() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbLayers();
    }

    //!
    //! \brief Serialize the network to a stream.
    //!
    //! \return A IHostMemory object that contains the serialized engine.
    //!
    //! The network may be deserialized with IRuntime::deserializeCudaEngine().
    //!
    //! \see IRuntime::deserializeCudaEngine()
    //!
    IHostMemory* serialize() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->serialize();
    }

    //!
    //! \brief Create an execution context and specify the strategy for allocating internal
    //! activation memory.
    //!
    //! The default value for the allocation strategy is
    //! ExecutionContextAllocationStrategy::kSTATIC, which means the context will pre-allocate a
    //! block of device memory that is sufficient for all profiles. The newly created execution
    //! context will be assigned optimization profile 0. If an error recorder has been set for the
    //! engine, it will also be passed to the execution context.
    //!
    //! \see IExecutionContext
    //! \see IExecutionContext::setOptimizationProfileAsync()
    //! \see ExecutionContextAllocationStrategy
    //!
    IExecutionContext*
    createExecutionContext(ExecutionContextAllocationStrategy strategy =
                               ExecutionContextAllocationStrategy::kSTATIC) noexcept
    {
        return new IExecutionContext{program_};
        // return mImpl->createExecutionContext(strategy);
    }

    //!
    //! \brief Get whether an input or output tensor must be on GPU or CPU.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return TensorLocation::kDEVICE if tensorName must be on GPU, or TensorLocation::kHOST if on
    //! CPU, or TensorLocation::kDEVICE if the provided name does not map to an input or output
    //! tensor.
    //!
    //! The location is established at build time. E.g. shape tensors inputs are typically required
    //! to be on the CPU.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    TensorLocation getTensorLocation(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorLocation(tensorName);
    }

    //!
    //! \brief True if tensor is required as input for shape calculations or is output from shape
    //! calculations.
    //!
    //! Return true for either of the following conditions:
    //!
    //! * The tensor is a network input, and its value is required for
    //! IExecutionContext::getTensorShape()
    //!   to return the shape of a network output.
    //!
    //! * The tensor is a network output, and inferShape() will compute its values.
    //!
    //! For example, if a network uses an input tensor "foo" as an addend to an IElementWiseLayer
    //! that computes the "reshape dimensions" for IShuffleLayer, then isShapeInferenceIO("foo") ==
    //! true. If the network copies said input tensor "foo" to an output "bar", then
    //! isShapeInferenceIO("bar") == true and IExecutionContext::inferShapes() will write to "bar".
    //!
    bool isShapeInferenceIO(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isShapeInferenceIO(tensorName);
    }

    //!
    //! \brief Determine whether a tensor is an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \return kINPUT if tensorName is an input, kOUTPUT if tensorName is an output, or kNONE if
    //! neither.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    TensorIOMode getTensorIOMode(char const* tensorName) const noexcept
    {
        return migraphx::contains(std::string(tensorName), "output") ? TensorIOMode::kOUTPUT
                                                                     : TensorIOMode::kINPUT;
        // return mImpl->getTensorIOMode(tensorName);
    }

    //!
    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by createExecutionContext() with
    //! parameter.
    //!
    [[deprecated]] IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->createExecutionContextWithoutDeviceMemory();
    }

    //!
    //! \brief Return the maximum device memory required by the context over all profiles.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    size_t getDeviceMemorySize() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDeviceMemorySize();
    }

    //!
    //! \brief Return the maximum device memory required by the context for a profile.
    //!
    //! \see IExecutionContext::setDeviceMemory()
    //!
    size_t getDeviceMemorySizeForProfile(int32_t profileIndex) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDeviceMemorySizeForProfile(profileIndex);
    }

    //!
    //! \brief Return true if an engine can be refit.
    //!
    //! \see nvinfer1::createInferRefitter()
    //!
    bool isRefittable() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isRefittable();
    }

    //!
    //! \brief Return the number of bytes per component of an element, or -1 if the provided name
    //! does not map to an input or output tensor.
    //!
    //! The vector component size is returned if getTensorVectorizedDim() != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator. \warning The function can only return the result of profile 0, and issues a
    //! warning message when there are multiple profiles in the engine, use
    //! getTensorBytesPerComponent with profileIndex when there are multiple profiles.
    //!
    //! \see getTensorVectorizedDim()
    //! \see getTensorBytesPerComponent(tensorName, profileIndex)
    //!
    int32_t getTensorBytesPerComponent(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorBytesPerComponent(tensorName);
    }

    //!
    //! \brief Return the number of bytes per component of an element of given profile, or -1 if the
    //! provided name does not map to an input or output tensor.
    //!
    //! The vector component size is returned if getTensorVectorizedDim(tensorName, profileIndex) !=
    //! -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see getTensorVectorizedDim(tensorName, profileIndex)
    //!
    int32_t getTensorBytesPerComponent(char const* tensorName, int32_t profileIndex) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorBytesPerComponentV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the number of components included in one element, or -1 if the provided name
    //! does not map to an input or output tensor.
    //!
    //! The number of elements in the vectors is returned if getTensorVectorizedDim() != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator. \warning The function can only return the result of profile 0, and issues a
    //! warning message when there are multiple profiles in the engine, use
    //! getTensorComponentsPerElement with profileIndex when there are multiple profiles.
    //!
    //! \see getTensorVectorizedDim()
    //! \see getTensorComponentsPerElement(tensorName, profileIndex)
    //!
    int32_t getTensorComponentsPerElement(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorComponentsPerElement(tensorName);
    }

    //!
    //! \brief Return the number of components included in one element of given profile, or -1 if
    //! the provided name does not map to an input or output tensor.
    //!
    //! The number of elements in the vectors is returned if getTensorVectorizedDim(tensorName,
    //! profileIndex) != -1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    //! \see getTensorVectorizedDim(tensorName, profileIndex)
    //!
    int32_t getTensorComponentsPerElement(char const* tensorName,
                                          int32_t profileIndex) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorComponentsPerElementV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the tensor format, or TensorFormat::kLINEAR if the provided name does not map
    //! to an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator. \warning This API can only return the tensor format of profile 0, and issues
    //! a warning message when there are multiple profiles in the engine, use getTensorFormat with
    //! profileIndex when there are multiple profiles.
    //!
    //! \see getTensorFormat(tensorName, profileIndex)
    //!
    TensorFormat getTensorFormat(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorFormat(tensorName);
    }

    //!
    //! \brief Return the tensor format of given profile, or TensorFormat::kLINEAR if the provided
    //! name does not map to an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query the format for.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    TensorFormat getTensorFormat(char const* tensorName, int32_t profileIndex) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorFormatV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the human readable description of the tensor format, or empty string if the
    //! provided name does not map to an input or output tensor.
    //!
    //! The description includes the order, vectorization, data type, and strides.
    //! Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row-major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two-wide channel vectorized row-major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator. \warning The function can only return the result of profile 0, and issues a
    //! warning message when there are multiple profiles in the engine, use getTensorFormatDesc with
    //! profileIndex when there are multiple profiles.
    //!
    char const* getTensorFormatDesc(char const* tensorName) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorFormatDesc(tensorName);
    }

    //!
    //! \brief Return the human readable description of the tensor format of given profile, or empty
    //! string if the provided name does not map to an input or output tensor.
    //!
    //! The description includes the order, vectorization, data type, and strides.
    //! Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row-major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two-wide channel vectorized row-major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param tensorName The name of an input or output tensor.
    //! \param profileIndex The profile index to query the format for.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    char const* getTensorFormatDesc(char const* tensorName, int32_t profileIndex) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorFormatDescV2(tensorName, profileIndex);
    }

    //!
    //! \brief Return the dimension index that the buffer is vectorized, or -1 if the provided name
    //! does not map to an input or output tensor.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator. \warning The function can only return the result of profile 0, and issues a
    //! warning message when there are
    //!  multiple profiles in the engine, use getTensorVectorizedDim with profileIndex when there
    //!  are multiple profiles.
    //!
    int32_t getTensorVectorizedDim(char const* tensorName) const noexcept
    {
        // TODO what?
        return -1;
        // return mImpl->getTensorVectorizedDim(tensorName);
    }

    //!
    //! \brief Return the dimension index that the buffer is vectorized of given profile, or -1 if
    //! the provided name does not map to an input or output tensor.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param tensorName The name of an input.
    //! \param profileIndex The profile index to query the format for.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    int32_t getTensorVectorizedDim(char const* tensorName, int32_t profileIndex) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTensorVectorizedDimV2(tensorName, profileIndex);
    }

    //!
    //! \brief Returns the name of the network associated with the engine.
    //!
    //! The name is set during network creation and is retrieved after
    //! building or deserialization.
    //!
    //! \see INetworkDefinition::setName(), INetworkDefinition::getName()
    //!
    //! \return A null-terminated C-style string representing the name of the network.
    //!
    char const* getName() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getName();
    }

    //!
    //! \brief Get the number of optimization profiles defined for this engine.
    //!
    //! \return Number of optimization profiles. It is always at least 1.
    //!
    //! \see IExecutionContext::setOptimizationProfileAsync()
    int32_t getNbOptimizationProfiles() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbOptimizationProfiles();
    }

    //!
    //! \brief Get the minimum / optimum / maximum dimensions for an input tensor given its name
    //! under an optimization profile.
    //!
    //! \param tensorName The name of an input tensor.
    //!
    //! \param profileIndex The profile index, which must be between 0 and
    //! getNbOptimizationProfiles()-1.
    //!
    //! \param select Whether to query the minimum, optimum, or maximum dimensions for this input
    //! tensor.
    //!
    //! \return The minimum / optimum / maximum dimensions for an input tensor in this profile.
    //!         If the profileIndex is invalid or provided name does not map to an input tensor,
    //!         return Dims{-1, {}}
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    Dims getProfileShape(char const* tensorName,
                         int32_t profileIndex,
                         OptProfileSelector select) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProfileShape(tensorName, profileIndex, select);
    }

    //!
    //! \brief Get the minimum / optimum / maximum values (not dimensions) for an input tensor given
    //! its name under an optimization profile. These correspond to the values set using
    //! IOptimizationProfile::setShapeValues when the engine was built.
    //!
    //! \param tensorName The name of an input tensor.
    //!
    //! \param profileIndex The profile index, which must be between 0 and
    //! getNbOptimizationProfiles()-1.
    //!
    //! \param select Whether to query the minimum, optimum, or maximum values for this input
    //! tensor.
    //!
    //! \return The minimum / optimum / maximum values for an input tensor in this profile.
    //!        If the profileIndex is invalid or the provided name does not map to an input tensor,
    //!        return nullptr.
    //!
    //! \warning The string tensorName must be null-terminated, and be at most 4096 bytes including
    //! the terminator.
    //!
    int32_t const* getProfileTensorValues(char const* tensorName,
                                          int32_t profileIndex,
                                          OptProfileSelector select) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProfileTensorValues(tensorName, profileIndex, select);
    }

    //!
    //! \brief Determine what execution capability this engine has.
    //!
    //! If the engine has EngineCapability::kSTANDARD, then all engine functionality is valid.
    //! If the engine has EngineCapability::kSAFETY, then only the functionality in safe engine is
    //! valid. If the engine has EngineCapability::kDLA_STANDALONE, then only serialize, destroy,
    //! and const-accessor functions are valid.
    //!
    //! \return The EngineCapability flag that the engine was built for.
    //!
    EngineCapability getEngineCapability() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getEngineCapability();
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during
    //! execution. This function will call incRefCount of the registered ErrorRecorder at least
    //! once. Setting recorder to nullptr unregisters the recorder with the interface, resulting in
    //! a call to decRefCount if a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //!
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        error_recorder_ = recorder;
        // return mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned
    //! if an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Query whether the engine was built with an implicit batch dimension.
    //!
    //! \return Always false since TensorRT 10.0 does not support an implicit batch dimension.
    //!
    //! \see createNetworkV2
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is no supported since TensorRT 10.0.
    //!
    [[deprecated]] bool hasImplicitBatchDimension() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->hasImplicitBatchDimension();
    }

    //!
    //! \brief return the tactic sources required by this engine.
    //!
    //! The value returned is equal to zero or more tactics sources set
    //! at build time via setTacticSources() in IBuilderConfig. Sources
    //! set by the latter but not returned by \ref ICudaEngine::getTacticSources
    //! do not reduce overall engine execution time, and can be removed from
    //! future builds to reduce build time.
    //!
    //! \see IBuilderConfig::setTacticSources()
    //!
    TacticSources getTacticSources() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTacticSources();
    }

    //!
    //! \brief Return the \ref ProfilingVerbosity the builder config was set to when the engine was
    //! built.
    //!
    //! \return the profiling verbosity the builder config was set to when the engine was built.
    //!
    //! \see IBuilderConfig::setProfilingVerbosity()
    //!
    ProfilingVerbosity getProfilingVerbosity() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProfilingVerbosity();
    }

    //!
    //! \brief Create a new engine inspector which prints the layer information in an engine or an
    //! execution context.
    //!
    //! \see IEngineInspector.
    //!
    IEngineInspector* createEngineInspector() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->createEngineInspector();
    }

    //!
    //! \brief Return number of IO tensors.
    //!
    //! It is the number of input and output tensors for the network from which the engine was
    //! built. The names of the IO tensors can be discovered by calling getIOTensorName(i) for i in
    //! 0 to getNbIOTensors()-1.
    //!
    //! \see getIOTensorName()
    //!
    int32_t getNbIOTensors() const noexcept
    {
        return tensor_names_.size();
        // return mImpl->getNbIOTensors();
    }

    //!
    //! \brief Return name of an IO tensor.
    //!
    //! \param index value between 0 and getNbIOTensors()-1
    //!
    //! \see getNbIOTensors()
    //!
    char const* getIOTensorName(int32_t index) const noexcept
    {
        return tensor_names_.at(index).c_str();
        // return mImpl->getIOTensorName(index);
    }

    //!
    //! \brief Return the hardware compatibility level of this engine.
    //!
    //! \return hardwareCompatibilityLevel The level of hardware
    //!        compatibility.
    //!
    //! This is only supported for Ampere and newer architectures.
    //!
    HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getHardwareCompatibilityLevel();
    }

    //!
    //! \brief Return the number of auxiliary streams used by this engine.
    //!
    //! This number will be less than or equal to the maximum allowed number of auxiliary streams
    //! set by IBuilderConfig::setMaxAuxStreams() API call when the engine was built.
    //!
    //! \return the number of auxiliary streams used by this engine.
    //!
    //! \see IBuilderConfig::setMaxAuxStreams(), IExecutionContext::setAuxStreams()
    //!
    int32_t getNbAuxStreams() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbAuxStreams();
    }

    //!
    //! \brief Create a serialization configuration object.
    //!
    //! \see ISerializationConfig
    //!
    ISerializationConfig* createSerializationConfig() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->createSerializationConfig();
    }

    //!
    //! \brief Serialize the network to a stream with the provided SerializationConfig.
    //!
    //! \return An IHostMemory object that contains the serialized engine.
    //!
    //! The network may be deserialized with IRuntime::deserializeCudaEngine().
    //!
    //! \see IRuntime::deserializeCudaEngine()
    //!
    IHostMemory* serializeWithConfig(ISerializationConfig& config) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->serializeWithConfig(config);
    }

    //!
    //! \brief Limit the maximum amount of GPU memory usable for network weights
    //! in bytes.
    //!
    //! \param gpuMemoryBudget  This parameter may take on 3 types of values:
    //!  -1: Allows TensorRT to choose the budget according to the streamable weights size.
    //!      Free CUDA memory will be queried at ::createExecutionContext and accordingly:
    //!       * If streamable weights all fit: weight streaming is not required and disabled.
    //!       * Otherwise: Budget is set to getMinimumWeightStreamingBudget
    //!   0: (default) Disables weight streaming. The execution may fail if the network is too large
    //!   for GPU memory.
    //!  >0: The maximum bytes of GPU memory that weights can occupy. It must be bounded by
    //!      [getMinimumWeightStreamingBudget, min(getStreamableWeightsSize - 1, free GPU memory)].
    //!
    //! By setting a weight limit, users can expect a GPU memory usage reduction
    //! of |network weights| - gpuMemoryBudget bytes. Maximum memory savings occur
    //! when gpuMemoryBudget is set to getMinimumWeightStreamingBudget.
    //!
    //! Streaming larger amounts of memory will likely result in lower performance
    //! except in some boundary cases where streaming weights allows the user to
    //! run larger batch sizes. The higher throughput offsets the increased
    //! latency in these cases. Tuning the value of the memory limit is
    //! recommended for best performance.
    //!
    //! \warning If weight streaming is active, then multiple concurrent IExecutionContexts will
    //! forced to run serially.
    //!
    //! \warning GPU memory for the weights is allocated upon the first IExecutionContext's creation
    //!          and deallocated upon the last one's destruction.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \return true if the memory limit is valid and the call was successful
    //!         otherwise false.
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING,
    //!      ICudaEngine::getWeightStreamingBudget
    //!      ICudaEngine::getMinimumWeightStreamingBudget,
    //!      ICudaEngine::getStreamableWeightsSize
    //!
    bool setWeightStreamingBudget(int64_t gpuMemoryBudget) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setWeightStreamingBudget(gpuMemoryBudget);
    }

    //!
    //! \brief Returns the current weight streaming device memory budget in bytes.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //! \returns The weight streaming budget in bytes. Please see ::setWeightStreamingBudget for the
    //! possible
    //!          values.
    //!
    //! \see BuilderFlag::kWEIGHT_STREAMING,
    //!      ICudaEngine::setWeightStreamingBudget,
    //!      ICudaEngine::getMinimumWeightStreamingBudget,
    //!      ICudaEngine::getStreamableWeightsSize
    //!
    int64_t getWeightStreamingBudget() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getWeightStreamingBudget();
    }

    //!
    //! \brief The minimum number of bytes of GPU memory required by network
    //! weights for successful weight streaming.
    //!
    //! This is a positive integer for engines with streamable weights because a
    //! staging buffer on the GPU is required to temporarily hold the streamed
    //! weights. The size of the staging buffer is determined by TensorRT and must
    //! be at least as large as the size of the largest streamable weight in the
    //! network.
    //!
    //! \warning BuilderFlag::kWEIGHT_STREAMING must be set during engine building.
    //!
    //!
    //! \returns The minimum number of bytes of GPU memory required for streaming.
    //!
    //! \see ICudaEngine::setWeightStreamingBudget
    //!
    int64_t getMinimumWeightStreamingBudget() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMinimumWeightStreamingBudget();
    }

    //!
    //! \brief Get the total size in bytes of all streamable weights.
    //!
    //! The set of streamable weights is a subset of all network weights. The
    //! total size may exceed free GPU memory.
    //!
    //! Returns 0 if BuilderFlag::kWEIGHT_STREAMING is unset during engine building.
    //!
    //!
    //! \returns The total size in bytes of all streamable weights.
    //!
    //! \see ICudaEngine::setWeightStreamingBudget
    //!
    int64_t getStreamableWeightsSize() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getStreamableWeightsSize();
    }

    //!
    //! \brief Check if a tensor is marked as a debug tensor.
    //!
    //! Determine whether the given name corresponds to a debug tensor.
    //!
    //! \returns True if tensor is a debug tensor, false otherwise.
    //!
    //! \see INetworkDefinition::markDebug
    //!
    bool isDebugTensor(char const* name) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isDebugTensor(name);
    }

    private:
    std::shared_ptr<migraphx::program> program_;
    std::vector<std::string> tensor_names_;
    IErrorRecorder* error_recorder_;

    // protected:
    //     apiv::VCudaEngine* mImpl;
};

//!
//! \class IRuntime
//!
//! \brief Allows a serialized functionally unsafe engine to be deserialized.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IRuntime : public INoCopy
{
    public:
    IRuntime(ILogger& logger) : logger_{logger} {}
    virtual ~IRuntime() noexcept = default;

    //!
    //! \brief Sets the DLA core used by the network. Defaults to -1.
    //!
    //! \param dlaCore The DLA core to execute the engine on, in the range [0,getNbDlaCores()).
    //!
    //! This function is used to specify which DLA core to use via indexing, if multiple DLA cores
    //! are available.
    //!
    //! \warning if getNbDLACores() returns 0, then this function does nothing.
    //!
    //! \see getDLACore()
    //!
    void setDLACore(int32_t dlaCore) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDLACore(dlaCore);
    }

    //!
    //! \brief Get the DLA core that the engine executes on.
    //!
    //! \return assigned DLA core or -1 for DLA not present or unset.
    //!
    int32_t getDLACore() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDLACore();
    }

    //!
    //! \brief Returns number of DLA hardware cores accessible or 0 if DLA is unavailable.
    //!
    int32_t getNbDLACores() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbDLACores();
    }

    //!
    //! \brief Set the GPU allocator.
    //!
    //! \param allocator Set the GPU allocator to be used by the runtime. All GPU memory acquired
    //! will use this allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! If nullptr is passed, the default allocator will be used.
    //!
    void setGpuAllocator(IGpuAllocator* allocator) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setGpuAllocator(allocator);
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during
    //! execution. This function will call incRefCount of the registered ErrorRecorder at least
    //! once. Setting recorder to nullptr unregisters the recorder with the interface, resulting in
    //! a call to decRefCount if a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        error_recorder_ = recorder;
        pass("Not Implemented", true);
        // mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A nullptr will be returned
    //! if an error handler has not been set.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Deserialize an engine from host memory.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the engine.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size) noexcept
    {
        std::shared_ptr<migraphx::program> program;
        try
        {
            program = std::make_shared<migraphx::program>(
                migraphx::load_buffer(reinterpret_cast<const char*>(blob), size));
        }
        catch(migraphx::exception e)
        {
            // TODO write to error recorder if set, otherwise to logger
            return nullptr;
        }
        auto* engine = new ICudaEngine{std::move(program)};
        if(error_recorder_)
            engine->setErrorRecorder(error_recorder_);

        return engine;
    }

    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! If an error recorder has been set for the runtime, it will also be passed to the
    //! engine.
    //!
    //! This deserialization path will reduce host memory usage when weight streaming is enabled.
    //!
    //! \param streamReader a read-only stream from which TensorRT will deserialize a
    //!        previously serialized engine.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    ICudaEngine* deserializeCudaEngine(IStreamReader& streamReader)
    {
        pass("Not Implemented", true);
        // return mImpl->deserializeCudaEngine(streamReader);
    }

    //!
    //! \brief get the logger with which the runtime was created
    //!
    //! \return the logger
    //!
    ILogger* getLogger() const noexcept { return &logger_; }

    //!
    //! \brief Set the maximum number of threads.
    //!
    //! \param maxThreads The maximum number of threads that can be used by the runtime.
    //! \return True if successful, false otherwise.
    //!
    //! The default value is 1 and includes the current thread.
    //! A value greater than 1 permits TensorRT to use multi-threaded algorithms.
    //! A value less than 1 triggers a kINVALID_ARGUMENT error.
    //!
    bool setMaxThreads(int32_t maxThreads) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setMaxThreads(maxThreads);
    }

    //!
    //! \brief Get the maximum number of threads that can be used by the runtime.
    //!
    //! Retrieves the maximum number of threads that can be used by the runtime.
    //!
    //! \return The maximum number of threads that can be used by the runtime.
    //!
    //! \see setMaxThreads()
    //!
    int32_t getMaxThreads() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMaxThreads();
    }

    //!
    //! \brief Set the directory that will be used by this runtime for temporary files.
    //!
    //! On some platforms the TensorRT runtime may need to create and use temporary files
    //! with read/write/execute permissions to implement runtime functionality.
    //!
    //! \param path Path to the temporary directory for use, or nullptr.
    //!
    //! If path is nullptr, then TensorRT will use platform-specific heuristics to pick
    //! a default temporary directory if required:
    //!
    //! - On UNIX/Linux platforms, TensorRT will first try the TMPDIR environment variable, then
    //! fall back to /tmp
    //! - On Windows, TensorRT will try the TEMP environment variable.
    //!
    //! See the TensorRT Developer Guide for more information.
    //!
    //! The default value is nullptr.
    //!
    //! \warning If path is not nullptr, it must be a non-empty string representing a relative
    //! or absolute path in the format expected by the host operating system.
    //!
    //! \warning The string path must be null-terminated, and be at most 4096 bytes including the
    //! terminator. Note that the operating system may have stricter path length requirements.
    //!
    //! \warning The process using TensorRT must have rwx permissions for the temporary directory,
    //! and the directory shall be configured to disallow other users from modifying created files
    //! (e.g. on Linux, if the directory is shared with other users, the sticky bit must be set).
    //!
    //! \see getTemporaryDirectory()
    //!
    void setTemporaryDirectory(char const* path) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setTemporaryDirectory(path);
    }

    //!
    //! \brief Get the directory that will be used by this runtime for temporary files.
    //!
    //! \returns A path to the temporary directory in use, or nullptr if no path is specified.
    //!
    //! \see setTemporaryDirectory()
    char const* getTemporaryDirectory() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTemporaryDirectory();
    }

    //!
    //! \brief Set the tempfile control flags for this runtime.
    //!
    //! \param flags The flags to set.
    //!
    //! The default value is all flags set, i.e.
    //!
    //! (1U << static_cast<uint32_t>(kALLOW_IN_MEMORY_FILES)) | (1U <<
    //! static_cast<uint32_t>(kALLOW_TEMPORARY_FILES))
    //!
    //! \see TempfileControlFlag, TempfileControlFlags, getTempfileControlFlags()
    //!
    void setTempfileControlFlags(TempfileControlFlags flags) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setTempfileControlFlags(flags);
    }

    //!
    //! \brief Get the tempfile control flags for this runtime.
    //!
    //! \return The flags currently set.
    //!
    //! \see TempfileControlFlag, TempfileControlFlags, setTempfileControlFlags()
    //!
    TempfileControlFlags getTempfileControlFlags() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTempfileControlFlags();
    }

    //!
    //! \brief Get the local plugin registry that can be used by the runtime.
    //!
    //! \return The local plugin registry that can be used by the runtime.
    //!
    IPluginRegistry& getPluginRegistry() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getPluginRegistry();
    }

    //!
    //! \brief Load IRuntime from the file.
    //!
    //! This method loads a runtime library from a shared library file. The runtime can then be used
    //! to execute a plan file built with BuilderFlag::kVERSION_COMPATIBLE and
    //! BuilderFlag::kEXCLUDE_LEAN_RUNTIME both set and built with the same version of TensorRT as
    //! the loaded runtime library.
    //!
    //! \param path Path to the runtime lean library.
    //!
    //! \return the runtime library, or nullptr if it could not be loaded
    //!
    //! \warning The path string must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    IRuntime* loadRuntime(char const* path) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->loadRuntime(path);
    }

    //!
    //! \brief Set whether the runtime is allowed to deserialize engines with host executable code.
    //!
    //! \param allowed Whether the runtime is allowed to deserialize engines with host executable
    //! code.
    //!
    //! The default value is false.
    //!
    void setEngineHostCodeAllowed(bool allowed) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setEngineHostCodeAllowed(allowed);
    }

    //!
    //! \brief Get whether the runtime is allowed to deserialize engines with host executable code.
    //!
    //! \return Whether the runtime is allowed to deserialize engines with host executable code.
    //!
    bool getEngineHostCodeAllowed() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getEngineHostCodeAllowed();
    }

    private:
    ILogger& logger_;
    IErrorRecorder* error_recorder_;

    // protected:
    //     apiv::VRuntime* mImpl;
};

//!
//! \brief Create an instance of an IRuntime class.
//!
//! \param logger The logging class for the runtime.
//!
inline IRuntime* createInferRuntime(ILogger& logger) noexcept { return new IRuntime{logger}; }

namespace safe {

using mgxinfer1::ICudaEngine;
using mgxinfer1::IRuntime;

} // namespace safe

//!
//! \brief Represents one or more NetworkDefinitionCreationFlag flags
//! using binary OR operations.
//!  e.g., 1U << NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
//!
//! \see IBuilder::createNetworkV2
//!
using NetworkDefinitionCreationFlags = uint32_t;

//!
//! \enum NetworkDefinitionCreationFlag
//!
//! \brief List of immutable network properties expressed at network creation time.
//! NetworkDefinitionCreationFlag is used with createNetworkV2() to specify immutable properties of
//! the network.
//!
//! \see IBuilder::createNetworkV2
//!
enum class NetworkDefinitionCreationFlag : int32_t
{
    //! Ignored because networks are always "explicit batch" in TensorRT 10.0.
    //!
    //! \deprecated Deprecated in TensorRT 10.0.
    kEXPLICIT_BATCH = 0,

    //! Mark the network to be strongly typed.
    //! Every tensor in the network has a data type defined in the network following only type
    //! inference rules and the inputs/operator annotations. Setting layer precision and layer
    //! output types is not allowed, and the network output types will be inferred based on the
    //! input types and the type inference rules.
    kSTRONGLY_TYPED = 1,
};

//!
//! \enum ActivationType
//!
//! \brief Enumerates the types of activation to perform in an activation layer.
//!
enum class ActivationType : int32_t
{
    kRELU             = 0,  //!< Rectified linear activation.
    kSIGMOID          = 1,  //!< Sigmoid activation.
    kTANH             = 2,  //!< TanH activation.
    kLEAKY_RELU       = 3,  //!< LeakyRelu activation: x>=0 ? x : alpha * x.
    kELU              = 4,  //!< Elu activation: x>=0 ? x : alpha * (exp(x) - 1).
    kSELU             = 5,  //!< Selu activation: x>0 ? beta * x : beta * (alpha*exp(x) - alpha)
    kSOFTSIGN         = 6,  //!< Softsign activation: x / (1+|x|)
    kSOFTPLUS         = 7,  //!< Parametric softplus activation: alpha*log(exp(beta*x)+1)
    kCLIP             = 8,  //!< Clip activation: max(alpha, min(beta, x))
    kHARD_SIGMOID     = 9,  //!< Hard sigmoid activation: max(0, min(1, alpha*x+beta))
    kSCALED_TANH      = 10, //!< Scaled tanh activation: alpha*tanh(beta*x)
    kTHRESHOLDED_RELU = 11, //!< Thresholded ReLU activation: x>alpha ? x : 0
    kGELU_ERF         = 12, //!< GELU erf activation: 0.5 * x * (1 + erf(sqrt(0.5) * x))
    kGELU_TANH =
        13 //!< GELU tanh activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))
};

//!
//! \brief Controls how shift, scale and power are applied in a Scale layer.
//!
//! \see IScaleLayer
//!
enum class ScaleMode : int32_t
{
    kUNIFORM     = 0, //!< Identical coefficients across all elements of the tensor.
    kCHANNEL     = 1, //!< Per-channel coefficients.
    kELEMENTWISE = 2  //!< Elementwise coefficients.
};

//!
//! \class Weights
//!
//! \brief An array of weights used as a layer parameter.
//!
//! When using the DLA, the cumulative size of all Weights used in a network
//! must be less than 512MB in size. If the build option kGPU_FALLBACK is specified,
//! then multiple DLA sub-networks may be generated from the single original network.
//!
//! The weights are held by reference until the engine has been built. Therefore the data referenced
//! by \p values field should be preserved until the build is complete.
//!
//! The term "empty weights" refers to Weights with weight coefficients ( \p count == 0 and \p
//! values == nullptr).
//!
class Weights
{
    public:
    DataType type;      //!< The type of the weights.
    void const* values; //!< The weight values, in a contiguous array.
    int64_t count;      //!< The number of weights in the array.
};

//!
//! \enum TopKOperation
//!
//! \brief Enumerates the operations that may be performed by a TopK layer.
//!
enum class TopKOperation : int32_t
{
    kMAX = 0, //!< Maximum of the elements.
    kMIN = 1, //!< Minimum of the elements.
};

//!
//! \brief Control form of IGatherLayer
//!
//! \see IGatherLayer
//!
enum class GatherMode : int32_t
{
    kDEFAULT = 0, //!< Similar to ONNX Gather
    kELEMENT = 1, //!< Similar to ONNX GatherElements
    kND      = 2  //!< Similar to ONNX GatherND
};

//!
//! \enum MatrixOperation
//!
//! \brief Enumerates the operations that may be performed on a tensor
//!        by IMatrixMultiplyLayer before multiplication.
//!
enum class MatrixOperation : int32_t
{
    //! Treat x as a matrix if it has two dimensions, or as a collection of
    //! matrices if x has more than two dimensions, where the last two dimensions
    //! are the matrix dimensions. x must have at least two dimensions.
    kNONE = 0,

    //! Like kNONE, but transpose the matrix dimensions.
    kTRANSPOSE = 1,

    //! Treat x as a vector if it has one dimension, or as a collection of
    //! vectors if x has more than one dimension. x must have at least one dimension.
    //!
    //! The first input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is
    //! equivalent to a tensor with dimensions [M, 1, K] with MatrixOperation::kNONE, i.e. is
    //! treated as M row vectors of length K, or dimensions [M, K, 1] with
    //! MatrixOperation::kTRANSPOSE.
    //!
    //! The second input tensor with dimensions [M,K] used with MatrixOperation::kVECTOR is
    //! equivalent to a tensor with dimensions [M, K, 1] with MatrixOperation::kNONE, i.e. is
    //! treated as M column vectors of length K, or dimensions [M, 1, K] with
    //! MatrixOperation::kTRANSPOSE.
    kVECTOR = 2,
};

//!
//! \enum PoolingType
//!
//! \brief The type of pooling to perform in a pooling layer.
//!
enum class PoolingType : int32_t
{
    kMAX = 0, //!< Maximum over elements
    kAVERAGE =
        1, //!< Average over elements. If the tensor is padded, the count includes the padding
    kMAX_AVERAGE_BLEND = 2 //!< Blending between max and average pooling: (1-blendFactor)*maxPool +
                           //!< blendFactor*avgPool
};

//!
//! \enum FillOperation
//!
//! \brief Enumerates the tensor fill operations that may performed by a fill layer.
//!
//! \see IFillLayer
//!
enum class FillOperation : int32_t
{
    //! Compute each value via an affine function of its indices.
    //! For example, suppose the parameters for the IFillLayer are:
    //!
    //! * Dimensions = [3,4]
    //! * Alpha = 1
    //! * Beta = [100,10]
    //!
    //! Element [i,j] of the output is Alpha + Beta[0]*i + Beta[1]*j.
    //! Thus the output matrix is:
    //!
    //!      1  11  21  31
    //!    101 111 121 131
    //!    201 211 221 231
    //!
    //! A static beta b is implicitly a 1D tensor, i.e. Beta = [b].
    kLINSPACE = 0,

    //! Randomly draw values from a uniform distribution.
    kRANDOM_UNIFORM = 1,

    //! Randomly draw values from a normal distribution.
    kRANDOM_NORMAL = 2
};

//!
//! \enum ScatterMode
//!
//! \brief Control form of IScatterLayer
//!
//! \see IScatterLayer
//!
enum class ScatterMode : int32_t
{
    kELEMENT = 0, //!< Similar to ONNX ScatterElements
    kND      = 1, //!< Similar to ONNX ScatterND
};

//!
//! \class INetworkDefinition
//!
//! \brief A network definition for input to the builder.
//!
//! A network definition defines the structure of the network, and combined with a IBuilderConfig,
//! is built into an engine using an IBuilder. An INetworkDefinition can have all dimensions
//! explicit, full dims mode, in the network definition. The former mode, i.e. the implicit batch
//! size mode, has been deprecated.
//!
//! A network with implicit batch dimensions returns the dimensions of a layer without the implicit
//! dimension, and instead the batch is specified at execute/enqueue time. If the network has all
//! dimensions specified, then the first dimension follows elementwise broadcast rules: if it is 1
//! for some inputs and is some value N for all other inputs, then the first dimension of each
//! output is N, and the inputs with 1 for the first dimension are broadcast. Having divergent batch
//! sizes across inputs to a layer is not supported.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class INetworkDefinition : public INoCopy
{
    public:
    virtual ~INetworkDefinition() noexcept = default;

    //!
    //! \brief Add an input tensor to the network.
    //!
    //! The name of the input tensor is used to find the index into the buffer array for an engine
    //! built from the network. The volume must be less than 2^31 elements.
    //!
    //! For networks with wildcard dimensions, the volume
    //! is based on the maxima specified by an IOptimizationProfile.Dimensions are normally
    //! non-negative integers. The exception is that in networks with all explicit dimensions, -1
    //! can be used as a wildcard for a dimension to be specified at runtime. Input tensors with
    //! such a wildcard must have a corresponding entry in the IOptimizationProfiles indicating the
    //! permitted extrema, and the input dimensions must be set by IExecutionContext::setInputShape.
    //! Different IExecutionContext instances can have different dimensions. Wildcard dimensions are
    //! only supported for EngineCapability::kSTANDARD. They are not supported in safety contexts.
    //! DLA does not support Wildcard dimensions.
    //!
    //! Tensor dimensions are specified independent of format.  For example, if a
    //! tensor is formatted in "NHWC" or a vectorized format, the dimensions are
    //! still specified in the order{N, C, H, W}. For 2D images with a channel
    //! dimension, the last three dimensions are always {C,H,W}. For 3D images
    //! with a channel dimension, the last four dimensions are always {C,D,H,W}.
    //!
    //! \param name The name of the tensor.
    //! \param type The type of the data held in the tensor.
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \warning It is an error to specify a wildcard value on a dimension that is determined by
    //! trained parameters.
    //!
    //! \warning If run on DLA with explicit dimensions, only leading dimension can be a wildcard.
    //! And provided profile must have same minimum, optimum, and maximum dimensions.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see ITensor
    //!
    //! \return The new tensor or nullptr if there is an error.
    //!
    ITensor* addInput(char const* name, DataType type, Dims const& dimensions) noexcept
    {
        auto* mm = program_->get_main_module();
        auto input =
            mm->add_parameter(name, migraphx::shape{fromDataType(type), dimsToVec(dimensions)});
        input_tensors_.push_back(std::make_unique<ITensor>(input));
        auto* ret = input_tensors_.back().get();
        return ret;
        // return mImpl->addInput(name, type, dimensions);
    }

    //!
    //! \brief Mark a tensor as a network output.
    //!
    //! \param tensor The tensor to mark as an output tensor.
    //!
    //! \warning It is an error to mark a network input as an output.
    //! \warning It is an error to mark a tensor inside an ILoop or an
    //!          IIfConditional as an output.
    //!
    void markOutput(ITensor& tensor) noexcept
    {
        pass("Not Implemented", false);
        // mImpl->markOutput(tensor);
    }

    //!
    //! \brief Mark a tensor as a debug tensor.
    //!
    //! A debug tensor can be optionally emitted at runtime.
    //! Note that tensor names are required to specify debug
    //! tensors at runtime.
    //!
    //! \param tensor Tensor to be marked as debug
    //!
    //! \return True if tensor successfully marked (or was already marked), false otherwise.
    //!
    //! \see unmarkDebug(), IExecutionContext::setDebugListener(), ITensor::setName()
    //!
    bool markDebug(ITensor& tensor) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->markDebug(tensor);
    }

    //!
    //! \brief Unmark a tensor as a debug tensor.
    //!
    //! Remove the marking of a tensor as a debug tensor.
    //!
    //! \param tensor Tensor to be unmarked as debug.
    //!
    //! \return True if tensor successfully unmarked (or was already unmarked), false otherwise.
    //!
    //! \see markDebug(), IExecutionContext::setDebugListener()
    //!
    bool unmarkDebug(ITensor& tensor) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->unmarkDebug(tensor);
    }

    //!
    //! \brief Check if a tensor is marked as debug tensor.
    //!
    //! \return true if tensor is marked as debug tensor, false otherwise.
    //!
    bool isDebugTensor(ITensor const& tensor) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isDebugTensor(tensor);
    }

    //!
    //! \brief Add an activation layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of activation function to apply.
    //!
    //! Note that the setAlpha() and setBeta() methods must be used on the
    //! output for activations that require these parameters.
    //!
    //! \see IActivationLayer ActivationType
    //!
    //! \warning Int32 and Int64 are valid only for activation type kRELU.
    //!
    //! \return The new activation layer, or nullptr if it could not be created.
    //!
    IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addActivation(input, type);
    }

    //!
    //! \brief Add a LRN layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param window The size of the window.
    //! \param alpha The alpha value for the LRN computation.
    //! \param beta The beta value for the LRN computation.
    //! \param k The k value for the LRN computation.
    //!
    //! \see ILRNLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new LRN layer, or nullptr if it could not be created.
    //!
    ILRNLayer* addLRN(ITensor& input, int64_t window, float alpha, float beta, float k) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addLRN(input, window, alpha, beta, k);
    }

    //!
    //! \brief Add a Scale layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!              This tensor must have at least 4 dimensions.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //!
    //! If the weights are available, then the size of weights are dependent on the ScaleMode.
    //! For ScaleMode::kUNIFORM, the number of weights equals 1.
    //! For ScaleMode::kCHANNEL, the number of weights equals the channel dimension.
    //! For ScaleMode::kELEMENTWISE, the number of weights equals the product of the last three
    //! dimensions of the input.
    //!
    //! \see addScaleNd
    //! \see IScaleLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new Scale layer, or nullptr if it could not be created.
    //!
    IScaleLayer*
    addScale(ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addScale(input, mode, shift, scale, power);
    }

    //!
    //! \brief Add a SoftMax layer to the network.
    //!
    //! \see ISoftMaxLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new SoftMax layer, or nullptr if it could not be created.
    //!
    ISoftMaxLayer* addSoftMax(ITensor& input) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addSoftMax(input);
    }

    //!
    //! \brief Add a concatenation layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //!
    //! \see IConcatenationLayer
    //!
    //! \return The new concatenation layer, or nullptr if it could not be created.
    //!
    //! \warning All tensors must have the same dimensions except along the concatenation axis.
    //!
    IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addConcatenation(inputs, nbInputs);
    }

    //!
    //! \brief Add an elementwise layer to the network.
    //!
    //! \param input1 The first input tensor to the layer.
    //! \param input2 The second input tensor to the layer.
    //! \param op The binary operation that the layer applies.
    //!
    //! The input tensors must have the same rank and compatible type.
    //! Two types are compatible if they are the same type or are both in the set {kFLOAT, kHALF}.
    //! For each dimension, their lengths must match, or one of them must be one.
    //! In the latter case, the tensor is broadcast along that axis.
    //!
    //! The output tensor has the same rank as the inputs.
    //! For each dimension, its length is the maximum of the lengths of the
    //! corresponding input dimension.
    //!
    //! The inputs are shape tensors if the output is a shape tensor.
    //!
    //! \see IElementWiseLayer
    //!
    //! \return The new elementwise layer, or nullptr if it could not be created.
    //!
    IElementWiseLayer*
    addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept
    {
        // auto mm = program_->get_main_module();

        // std::string elem_wise_op;
        // switch(op)
        // {
        // case ElementWiseOperation::kSUM: elem_wise_op = "add"; break;
        // default: pass("Not Implemented", true);
        // }

        // auto elem_wise_ins = mm->add_instruction(
        //     migraphx::make_op(elem_wise_op), input1.getInstruction(), input2.getInstruction());

        // std::vector<migraphx::instruction_ref> input_instructions{elem_wise_ins};
        // std::vector<migraphx::instruction_ref> output_instructions{elem_wise_ins};

        // layers_.push_back(
        //     std::make_unique<IElementWiseLayer>(program_, input_instructions,
        //     output_instructions));
        // return dynamic_cast<IElementWiseLayer*>(layers_.back().get());
        // return mImpl->addElementWise(input1, input2, op);
    }

    //!
    //! \brief Add a unary layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The operation to apply.
    //!
    //! \see IUnaryLayer
    //!
    //! Generally the input must have a floating-point type (or kINT8 as a quantized float),
    //! except for the following operations:
    //! * kSIGN accepts a floating-point or Int32 tensor.
    //! * kNOT requires a Bool tensor.
    //!
    //! The input is a shape tensor if the output is a shape tensor.
    //!
    //! \return The new unary layer, or nullptr if it could not be created
    //!

    IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept
    {

        // std::string unary_op = trtUnaryOperationToMGXOp(operation);

        // auto unary_ins = mm->add_instruction(migraphx::make_op(unary_op),
        // input.getInstruction());

        // std::vector<ITensor*> inputs{&input};
        // std::vector<migraphx::instruction_ref> outputs{unary_ins};
        layers_.push_back(std::make_unique<IUnaryLayer>(input, operation, program_));
        return dynamic_cast<IUnaryLayer*>(layers_.back().get());
        // return mImpl->addUnary(input, operation);
    }

    //!
    //! \brief Add a shuffle layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShuffleLayer
    //!
    //! \return The new shuffle layer, or nullptr if it could not be created.
    //!
    IShuffleLayer* addShuffle(ITensor& input) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addShuffle(input);
    }

    //!
    //! \brief Add a OneHot layer to the network.
    //!
    //! \param indices - tensor containing indices where on_value should be set.
    //! \param values - a 2-element tensor, consisting of [off_value, on_value].
    //! \param depth - a shape tensor containing the width of the added one-hot dimension.
    //! \param axis - the axis to add the one-hot encoding to.
    //!
    //! \see IOneHotLayer
    //!
    //! \return The new OneHot layer, or nullptr if it could not be created.
    //!
    IOneHotLayer*
    addOneHot(ITensor& indices, ITensor& values, ITensor& depth, int32_t axis) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addOneHot(indices, values, depth, axis);
    }

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! \return The number of layers in the network.
    //!
    //! \see getLayer()
    //!
    int32_t getNbLayers() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbLayers();
    }

    //!
    //! \brief Get the layer specified by the given index.
    //!
    //! \param index The index of the layer.
    //!
    //! \return The layer, or nullptr if the index is out of range.
    //!
    //! \see getNbLayers()
    //!
    ILayer* getLayer(int32_t index) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getLayer(index);
    }

    //!
    //! \brief Get the number of inputs in the network.
    //!
    //! \return The number of inputs in the network.
    //!
    //! \see getInput()
    //!
    int32_t getNbInputs() const noexcept
    {
        // TODO store parameter_names and parameter_shapes as data members;
        return program_->get_parameter_names().size();
        // return mImpl->getNbInputs();
    }

    //!
    //! \brief Get the input tensor specified by the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range.
    //!
    //! \note adding inputs invalidates indexing here
    //!
    //! \see getNbInputs()
    //!
    ITensor* getInput(int32_t index) const noexcept
    {
        // TODO check that index is in range
        // auto param_names = program_->get_parameter_names();
        // Paired with the usage in mnist_sample.cpp, this currently causes a memory leak. One must
        // suppose that the network owns the underlying memory, and just passes back a pointer to
        // the ITensor.
        // auto ins = program_->get_parameter(param_names.at(index));
        return input_tensors_[index].get();
        // return mImpl->getInput(index);
    }

    //!
    //! \brief Get the number of outputs in the network.
    //!
    //! The outputs include those marked by markOutput or markOutputForShapes.
    //!
    //! \return The number of outputs in the network.
    //!
    //! \see getOutput()
    //!
    int32_t getNbOutputs() const noexcept
    {
        return program_->get_output_shapes().size();
        // return mImpl->getNbOutputs();
    }

    //!
    //! \brief Get the output tensor specified by the given index.
    //!
    //! \param index The index of the output tensor.
    //!
    //! \return The output tensor, or nullptr if the index is out of range.
    //!
    //! \note adding inputs invalidates indexing here
    //!
    //! \see getNbOutputs()
    //!
    ITensor* getOutput(int32_t index) const noexcept
    {
        // Paired with the usage in mnist_sample.cpp, this currently causes a memory leak. One must
        // suppose that the network owns the underlying memory, and just passes back a pointer to
        // the ITensor.
        // TODO is this correct?
        // auto ins = program_->get_main_module()->get_returns()[index];
        return output_tensors_[index].get();
        // pass("Not Implemented", true);
        // return mImpl->getOutput(index);
    }

    //!
    //! \brief Add a reduce layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The reduction operation to perform.
    //! \param reduceAxes The reduction dimensions.
    //!        The bit in position i of bitmask reduceAxes corresponds to explicit dimension i if
    //!        result. E.g., the least significant bit corresponds to the first explicit dimension
    //!        and the next to least significant bit corresponds to the second explicit dimension.
    //! \param keepDimensions The boolean that specifies whether or not to keep the reduced
    //! dimensions in the output of the layer.
    //!
    //! The reduce layer works by performing an operation specified by \p operation to reduce the
    //! tensor \p input across the axes specified by \p reduceAxes.
    //!
    //! \see IReduceLayer
    //!
    //! \warning If output is an Int32 or Int64 shape tensor, ReduceOperation::kAVG is unsupported.
    //!
    //! \return The new reduce layer, or nullptr if it could not be created.
    //!
    IReduceLayer* addReduce(ITensor& input,
                            ReduceOperation operation,
                            uint32_t reduceAxes,
                            bool keepDimensions) noexcept
    {
        auto mm = program_->get_main_module();

        std::string reduce_op;
        switch(operation)
        {
        case ReduceOperation::kSUM: reduce_op = "reduce_sum"; break;
        case ReduceOperation::kPROD: reduce_op = "reduce_prod"; break;
        case ReduceOperation::kMAX: reduce_op = "reduce_max"; break;
        case ReduceOperation::kMIN: reduce_op = "reduce_min"; break;
        case ReduceOperation::kAVG: reduce_op = "reduce_mean"; break;
        }

        std::vector<int64_t> axes = axesToVector(reduceAxes);

        auto reduce_ins = mm->add_instruction(migraphx::make_op(reduce_op, {{"axes", axes}}),
                                              input.getInstruction());

        if(!keepDimensions)
        {
            reduce_ins =
                mm->add_instruction(migraphx::make_op("squeeze", {{"axes", axes}}), reduce_ins);
        }

        std::vector<ITensor*> inputs{&input};
        std::vector<migraphx::instruction_ref> outputs{reduce_ins};
        layers_.push_back(std::make_unique<IReduceLayer>(operation, program_));
        return dynamic_cast<IReduceLayer*>(layers_.back().get());
        //  return mImpl->addReduce(input, operation, reduceAxes, keepDimensions);
    }

    //!
    //! \brief Add a TopK layer to the network.
    //!
    //! The TopK layer has two outputs of the same dimensions. The first contains data values,
    //! the second contains index positions for the values. Output values are sorted, largest first
    //! for operation kMAX and smallest first for operation kMIN.
    //!
    //! Currently only values of K up to 3840 are supported.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \param op Operation to perform.
    //!
    //! \param k The number of elements to keep. For dynamic k, use the setInput() method to pass in
    //! k as a tensor
    //!        instead, which will override the static k value passed here in calculations.
    //!
    //! \param reduceAxes The reduction dimensions.
    //!        The bit in position i of bitmask reduceAxes corresponds to explicit dimension i of
    //!        the result. E.g., the least significant bit corresponds to the first explicit
    //!        dimension and the next to least significant bit corresponds to the second explicit
    //!        dimension.
    //!
    //!        Currently reduceAxes must specify exactly one dimension, and it must be one of the
    //!        last four dimensions.
    //!
    //! \see ITopKLayer
    //!
    //! \return The new TopK layer, or nullptr if it could not be created.
    //!
    ITopKLayer* addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addTopK(input, op, k, reduceAxes);
    }

    //!
    //! \brief Add gather with mode GatherMode::kDEFAULT and specified axis and nbElementWiseDims=0.
    //!
    //! \param data The tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param axis The axis in the data tensor to gather on.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    IGatherLayer* addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addGather(data, indices, axis);
    }

    //!
    //! \brief Add gather with specified mode, axis=0 and nbElementWiseDims=0.
    //!
    //! \param data The tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param mode The gather mode.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    IGatherLayer* addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addGatherV2(data, indices, mode);
    }

    //!
    //! \brief Add a RaggedSoftMax layer to the network.
    //!
    //! \param input The ZxS input tensor.
    //! \param bounds The Zx1 bounds tensor.
    //!
    //! \see IRaggedSoftMaxLayer
    //!
    //! \warning The bounds tensor cannot have the last dimension be the wildcard character.
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning The input and bounds tensors should be 3D tensors.
    //!
    //! \return The new RaggedSoftMax layer, or nullptr if it could not be created.
    //!
    IRaggedSoftMaxLayer* addRaggedSoftMax(ITensor& input, ITensor& bounds) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addRaggedSoftMax(input, bounds);
    }

    //!
    //! \brief Add a MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor (commonly A).
    //! \param op0 The operation to apply to input0.
    //! \param input1 The second input tensor (commonly B).
    //! \param op1 The operation to apply to input1.
    //!
    //! The inputs are shape tensors if the output is a shape tensor.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    IMatrixMultiplyLayer* addMatrixMultiply(ITensor& input0,
                                            MatrixOperation op0,
                                            ITensor& input1,
                                            MatrixOperation op1) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addMatrixMultiply(input0, op0, input1, op1);
    }

    //!
    //! \brief Add a nonzero layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see INonZeroLayer
    //!
    //! \return The new nonzero layer, or nullptr if it could be created.
    //!
    INonZeroLayer* addNonZero(ITensor& input) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addNonZero(input);
    }

    //!
    //! \brief Add a constant layer to the network.
    //!
    //! \param dimensions The dimensions of the constant.
    //! \param weights The constant value, represented as weights.
    //!
    //! \see IConstantLayer
    //!
    //! \return The new constant layer, or nullptr if it could not be created.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! If a wildcard dimension is used, the volume of the runtime dimensions must equal
    //! the number of weights specified.
    //!
    //! \warning DataType::kUINT8 not supported.
    //!
    IConstantLayer* addConstant(Dims const& dimensions, Weights weights) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addConstant(dimensions, weights);
    }

    //!
    //! \brief Add an identity layer.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IIdentityLayer
    //!
    //! \return The new identity layer, or nullptr if it could not be created.
    //!
    IIdentityLayer* addIdentity(ITensor& input) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addIdentity(input);
    }

    //!
    //! \brief Add a cast layer.
    //!
    //! \param input The input tensor to the layer.
    //! \param toType The DataType of the output tensor
    //!
    //! \see ICastLayer
    //!
    //! \return The new cast layer, or nullptr if it could not be created.
    //!
    ICastLayer* addCast(ITensor& input, DataType toType) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addCast(input, toType);
    }

    //!
    //! \brief remove a tensor from the network definition.
    //!
    //! \param tensor the tensor to remove
    //!
    //! It is illegal to remove a tensor that is the input or output of a layer.
    //! if this method is called with such a tensor, a warning will be emitted on the log
    //! and the call will be ignored. Its intended use is to remove detached tensors after
    //! e.g. concatenating two networks with Layer::setInput().
    //!
    void removeTensor(ITensor& tensor) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->removeTensor(tensor);
    }

    //!
    //! \brief unmark a tensor as a network output.
    //!
    //! \param tensor The tensor to unmark as an output tensor.
    //!
    //! see markOutput()
    //!
    void unmarkOutput(ITensor& tensor) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->unmarkOutput(tensor);
    }

    //!
    //! \brief Add a plugin layer to the network using the IPluginV2 interface.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginV2Layer
    //!
    //! \warning Dimension wildcard are only supported with IPluginV2DynamicExt or IPluginV2IOExt
    //! plugins. \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    IPluginV2Layer*
    addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addPluginV2(inputs, nbInputs, plugin);
    }

    //!
    //! \brief Add a plugin layer implementing the IPluginV3 interface to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param shapeInputs Shape tensor inputs to the layer.
    //! \param nbShapeInputs The number of shape tensor inputs.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginV3Layer
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    IPluginV3Layer* addPluginV3(ITensor* const* inputs,
                                int32_t nbInputs,
                                ITensor* const* shapeInputs,
                                int32_t nbShapeInputs,
                                IPluginV3& plugin) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addPluginV3(inputs, nbInputs, shapeInputs, nbShapeInputs, plugin);
    }

    //!
    //! \brief Add a slice layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param start The start offset
    //! \param size The output dimension
    //! \param stride The slicing stride
    //!
    //! Positive, negative, zero stride values, and combinations of them in different dimensions are
    //! allowed.
    //!
    //! \see ISliceLayer
    //!
    //! \return The new slice layer, or nullptr if it could not be created.
    //!
    ISliceLayer*
    addSlice(ITensor& input, Dims const& start, Dims const& size, Dims const& stride) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addSlice(input, start, size, stride);
    }

    //!
    //! \brief Sets the name of the network.
    //!
    //! \param name The name to assign to this network.
    //!
    //! Set the name of the network so that it can be associated with a built
    //! engine. The \p name must be a null-terminated C-style string.
    //! TensorRT makes no use of this string except storing it as part of the engine
    //! so that it may be retrieved at runtime.
    //! A name unique to the builder will be generated by default.
    //!
    //! This method copies the name string.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    //! \see INetworkDefinition::getName(), ISafeCudaEngine::getName()
    //!
    //! \return none
    //!
    void setName(char const* name) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setName(name);
    }

    //!
    //! \brief Returns the name associated with the network.
    //!
    //! The memory pointed to by getName() is owned by the INetworkDefinition object.
    //!
    //! \see INetworkDefinition::setName()
    //!
    //! \return A null-terminated C-style string representing the name of the network.
    //!
    char const* getName() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getName();
    }

    //!
    //! \brief Add a shape layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShapeLayer
    //!
    //! \warning addShape is only supported when hasImplicitBatchDimensions is false.
    //!
    //! \return The new shape layer, or nullptr if it could not be created.
    //!
    IShapeLayer* addShape(ITensor& input) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addShape(input);
    }

    //!
    //! \brief Query whether the network was created with an implicit batch dimension.
    //!
    //! \return Always false since TensorRT 10.0 does not support an implicit batch dimension.
    //!
    //! \see createNetworkV2
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Implicit batch is not supported since
    //! TensorRT 10.0.
    //!
    [[deprecated]] bool hasImplicitBatchDimension() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->hasImplicitBatchDimension();
    }

    //!
    //! \brief Get the network definition creation flags for this network definition object.
    //! Defaults to 0.
    //!
    //! \return The network definition creation options as a bitmask.
    //!
    NetworkDefinitionCreationFlags getFlags() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getFlags();
    }

    //!
    //! \brief Returns true if the network definition creation flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    bool getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getFlag(networkDefinitionCreationFlag);
    }

    //!
    //! \brief Enable tensor's value to be computed by IExecutionContext::getShapeBinding.
    //!
    //! \return True if successful, false if tensor is already marked as an output.
    //!
    //! The tensor must be of type DataType::kINT32 and have no more than one dimension.
    //!
    //! \warning The tensor must have dimensions that can be determined to be constants at build
    //! time.
    //!
    //! \warning It is an error to mark a network input as a shape output.
    //!
    //!
    bool markOutputForShapes(ITensor& tensor) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->markOutputForShapes(tensor);
    }

    //!
    //! \brief Undo markOutputForShapes.
    //!
    //! \warning inputs to addShape cannot contain wildcard dimension values.
    //!
    //! \return True if successful, false if tensor is not marked as an output.
    //!
    bool unmarkOutputForShapes(ITensor& tensor) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->unmarkOutputForShapes(tensor);
    }

    //!
    //! \brief Add a parametric ReLU layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param slope The slope tensor to the layer. This tensor should be unidirectionally
    //! broadcastable
    //!        to the input tensor.
    //!
    //! \see IParametricReLULayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new parametric ReLU layer, or nullptr if it could not be created.
    //!
    IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addParametricReLU(input, slope);
    }

    //!
    //! \brief Add a multi-dimension convolution layer to the network.
    //!
    //! \param input The input tensor to the convolution.
    //! \param nbOutputMaps The number of output feature maps for the convolution.
    //! \param kernelSize The multi-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The bias weights for the convolution. Weights{} represents no bias.
    //!
    //! \see IConvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input
    //! tensor. \warning Int32 tensors are not valid input tensors. \warning Only 2D or 3D
    //! convolution is supported.
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    IConvolutionLayer* addConvolutionNd(ITensor& input,
                                        int64_t nbOutputMaps,
                                        Dims const& kernelSize,
                                        Weights kernelWeights,
                                        Weights biasWeights) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addConvolutionNd(input, nbOutputMaps, kernelSize, kernelWeights,
        // biasWeights);
    }

    //!
    //! \brief Add a multi-dimension pooling layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of pooling to apply.
    //! \param windowSize The size of the pooling window.
    //!
    //! \see IPoolingLayer PoolingType
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D or 3D pooling is supported.
    //!
    //! \return The new pooling layer, or nullptr if it could not be created.
    //!
    IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims const& windowSize) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addPoolingNd(input, type, windowSize);
    }

    //!
    //! \brief Add a multi-dimension deconvolution layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputMaps The number of output feature maps.
    //! \param kernelSize The multi-dimensions of the deconvolution kernel.
    //! \param kernelWeights The kernel weights for the deconvolution.
    //! \param biasWeights The bias weights for the deconvolution. Weights{} represents no bias.
    //!
    //! \see IDeconvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input
    //! tensor. \warning Int32 tensors are not valid input tensors. \warning Only 2D or 3D
    //! deconvolution is supported.
    //
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    IDeconvolutionLayer* addDeconvolutionNd(ITensor& input,
                                            int64_t nbOutputMaps,
                                            Dims kernelSize,
                                            Weights kernelWeights,
                                            Weights biasWeights) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addDeconvolutionNd(input, nbOutputMaps, kernelSize, kernelWeights,
        // biasWeights);
    }

    //!
    //! \brief Add a multi-dimension scale layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //! \param channelAxis The channel axis.
    //!
    //! If the weights are available, then the size of weights are dependent on the ScaleMode.
    //! For ScaleMode::kUNIFORM, the number of weights equals 1.
    //! For ScaleMode::kCHANNEL, the number of weights equals the channel dimension.
    //! For ScaleMode::kELEMENTWISE, the number of weights equals the product of all input
    //! dimensions at channelAxis and beyond.
    //!
    //! For example, if the inputs dimensions are [A,B,C,D,E,F], and channelAxis=2:
    //! For ScaleMode::kUNIFORM, the number of weights is equal to 1.
    //! For ScaleMode::kCHANNEL, the number of weights is C.
    //! For ScaleMode::kELEMENTWISE, the number of weights is C*D*E*F.
    //!
    //! channelAxis can also be set explicitly using setChannelAxis().
    //!
    //! \see IScaleLayer
    //! \see setChannelAxis()
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //! \warning Only 2D or 3D scale is supported.
    //!
    //! \return The new Scale layer, or nullptr if it could not be created.
    //!
    IScaleLayer* addScaleNd(ITensor& input,
                            ScaleMode mode,
                            Weights shift,
                            Weights scale,
                            Weights power,
                            int32_t channelAxis) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addScaleNd(input, mode, shift, scale, power, channelAxis);
    }

    //!
    //! \brief Add a resize layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IResizeLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new resize layer, or nullptr if it could not be created.
    //!
    IResizeLayer* addResize(ITensor& input) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addResize(input);
    }

    //!
    //! \brief Add a loop to the network.
    //!
    //! An ILoop provides a way to specify a recurrent subgraph.
    //!
    //! \return Pointer to ILoop that can be used to add loop-boundary layers for the loop.
    //!
    //! \see ILoop
    //!
    ILoop* addLoop() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addLoop();
    }

    //!
    //! \brief Add an if-then-else to the network.
    //!
    //! An IIfConditional provides a way to conditionally execute parts of the network.
    //!
    //! \return Pointer to the IIfConditional that can be used to add conditional-boundary layers
    //!         for the if-then-else.
    //!
    //! \see IIfConditional
    //!
    IIfConditional* addIfConditional() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addIfConditional();
    }

    //!
    //! \brief Add a select layer to the network.
    //!
    //! \param condition The condition tensor to the layer. Must have type DataType::kBOOL.
    //! \param thenInput The "then" input tensor to the layer.
    //! \param elseInput The "else" input tensor to the layer.
    //!
    //! All three input tensors must have the same rank, and along each axis
    //! must have the same length or a length of one. If the length is one, the tensor
    //! is broadcast along that axis. The output tensor has the dimensions of the inputs AFTER
    //! the broadcast rule is applied. For example, given:
    //!
    //!    dimensions of condition:  [1,1,5,9]
    //!    dimensions of thenInput:  [1,1,5,9]
    //!    dimensions of elseInput:  [1,3,1,9]
    //!
    //! the output dimensions are [1,3,5,9], and the output contents are defined by:
    //!
    //!      output[0,i,j,k] = condition[0,0,j,k] ? thenInput[0,0,j,k] : elseInput[0,i,0,k]
    //!
    //! The output dimensions are not necessarily the max of the input dimensions if any input
    //! is an empty tensor. For example, if in the preceding example, 5 is changed to 0:
    //!
    //!    dimensions of condition:  [1,1,0,9]
    //!    dimensions of thenInput:  [1,1,0,9]
    //!    dimensions of elseInput:  [1,3,1,9]
    //!
    //! then the output dimensions are [1,3,0,9].
    //!
    //! The inputs are shape tensors if the output is a shape tensor.
    //!
    //! \see ISelectLayer
    //!
    //! \return The new select layer, or nullptr if it could not be created.
    ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addSelect(condition, thenInput, elseInput);
    }

    //!
    //! \brief Add an assertion layer to the network.
    //!
    //! \param condition The input tensor to the layer.
    //! \param message A message to print if the assertion fails.
    //!
    //! \see IAssertionLayer
    //!
    //! \return The new assertion layer, or nullptr if it could not be created.
    //!
    //! The input tensor must be a boolean shape tensor.
    //!
    IAssertionLayer* addAssertion(ITensor& condition, char const* message) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addAssertion(condition, message);
    }

    //!
    //! \brief Add a fill layer to the network.
    //!
    //! \param dimensions The output tensor dimensions if input 0 is missing.
    //! \param op The fill operation that the layer applies.
    //!
    //! \warning For FillOperation::kLINSPACE, dimensions.nbDims must be 1 for static start/delta.
    //! If delta is provided as a 1D tensor, the length of delta must match dimensions.nbDims.
    //!
    //! This layer is non-deterministic across subsequent calls as the same inputs will produce
    //! different output tensors if \p op is either FillOperation::kRANDOM_UNIFORM or
    //! FillOperation::kRANDOM_NORMAL due to random state being shared across calls. The output
    //! tensors generated are determinstic when starting from the same initial state.
    //!
    //! \see IFillLayer
    //!
    //! \return The new fill layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 9.0. Superseded by three-argument addFill.
    //!
    [[deprecated]] IFillLayer* addFill(Dims const& dimensions, FillOperation op) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addFill(dimensions, op);
    }

    //!
    //! \brief Add a fill layer to the network.
    //!
    //! \param dimensions The output tensor dimensions if input 0 is missing.
    //! \param op The fill operation that the layer applies.
    //! \param outputType Optional output tensor data type, must be DataType::kFLOAT,
    //! DataType::kHALF, DataType::kINT32, or DataType::kINT64. This parameter is only used for
    //! static alpha/beta. Future calls to set output type using setToType or setOutputType must be
    //! consistent.
    //!
    //! \warning For FillOperation::kLINSPACE, dimensions.nbDims must be 1 for static start/delta.
    //! If delta is provided as a 1D tensor, the length of delta must match dimensions.nbDims.
    //!
    //! This layer is non-deterministic across subsequent calls as the same inputs will produce
    //! different output tensors if \p op is either FillOperation::kRANDOM_UNIFORM or
    //! FillOperation::kRANDOM_NORMAL due to random state being shared across calls. The output
    //! tensors generated are deterministic when starting from the same initial state.
    //!
    //! \see IFillLayer
    //!
    //! \return The new fill layer, or nullptr if it could not be created.
    //!
    IFillLayer* addFill(Dims const& dimensions, FillOperation op, DataType outputType) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addFillV2(dimensions, op, outputType);
    }

    //!
    //! \brief Add a padding layer to the network. Only 2D padding is currently supported.
    //!
    //! \param input The input tensor to the layer.
    //! \param prePadding The padding to apply to the start of the tensor.
    //! \param postPadding The padding to apply to the end of the tensor.
    //!
    //! \see IPaddingLayer
    //!
    //! \return The new padding layer, or nullptr if it could not be created.
    //!
    IPaddingLayer*
    addPaddingNd(ITensor& input, Dims const& prePadding, Dims const& postPadding) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addPaddingNd(input, prePadding, postPadding);
    }

    //!
    //! \brief Associate a name with all current uses of the given weights.
    //!
    //! The name must be set after the Weights are used in the network.
    //! Lookup is associative. The name applies to all Weights with matching
    //! type, value pointer, and count. If Weights with a matching value
    //! pointer, but different type or count exists in the network, an
    //! error message is issued, the name is rejected, and return false.
    //! If the name has already been used for other weights,
    //! return false. A nullptr causes the weights to become unnamed,
    //! i.e. clears any previous name.
    //!
    //! \param weights The weights to be named.
    //! \param name The name to associate with the weights.
    //!
    //! \return true on success.
    //!
    //! \warning The string name must be null-terminated, and be at most 4096 bytes including the
    //! terminator.
    //!
    bool setWeightsName(Weights weights, char const* name) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setWeightsName(weights, name);
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during
    //! execution. This function will call incRefCount of the registered ErrorRecorder at least
    //! once. Setting recorder to nullptr unregisters the recorder with the interface, resulting in
    //! a call to decRefCount if a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class.
    //! A nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Add a dequantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IDequantizeLayer
    //!
    //! \p input tensor data type must be DataType::kINT8/DataType::kFP8.
    //! \p scale tensor data type must be DataType::kFLOAT. The subgraph which terminates with the
    //! \p scale tensor must be a build-time constant.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 9.0. Superseded by three-argument addDequantize.
    //!
    [[deprecated]] IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addDequantize(input, scale);
    }

    //!
    //! \brief Add a dequantization layer to the network.
    //!
    //! \param input The input tensor to be dequantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IDequantizeLayer
    //!
    //! \p input tensor data type must be DataType::kINT8/DataType::kFP8/DataType::kINT4.
    //! \p scale tensor data type defaults to DataType::kFLOAT. For strongly typed networks, it must
    //! be the same as the output data type. The subgraph which terminates with the \p scale tensor
    //! must be a build-time constant. \p outputType output tensor data type, default value is
    //! DataType::kFLOAT. Future calls to set output type using setToType or setOutputType must be
    //! consistent. For strongly typed networks, it must be the same as the scale data type.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale, DataType outputType) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addDequantizeV2(input, scale, outputType);
    }

    //!
    //! \brief Add a Scatter layer to the network with specified mode and axis=0.
    //!
    //! \param data The input tensor to be updated with additional values.
    //! \param indices indices of the elements to be updated.
    //! \param updates values to be used for updates.
    //! \param mode scatter mode.
    //!
    //! \see IScatterLayer
    //!
    //! \p indices tensor data type must be DataType::kINT32.
    //! \p updates tensor data type must be the same as \p data
    //!
    //! \return The new Scatter layer, or nullptr if it could not be created.
    //!
    IScatterLayer*
    addScatter(ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addScatter(data, indices, updates, mode);
    }

    //!
    //! \brief Add a quantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IQuantizeLayer
    //!
    //! \p input tensor data type must be DataType::kFLOAT/DataType::kHALF.
    //! \p scale tensor data type must be DataType::kFLOAT. The subgraph which terminates with the
    //! \p scale tensor must be a build-time constant.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    //! \deprecated Deprecated in TensorRT 9.0. Superseded by three-argument addQuantize.
    //!
    [[deprecated]] IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addQuantize(input, scale);
    }

    //!
    //! \brief Add a quantization layer to the network.
    //!
    //! \param input The input tensor to be quantized.
    //! \param scale A tensor with the scale value.
    //!
    //! \see IQuantizeLayer
    //!
    //! \p input tensor data type must be DataType::kFLOAT/DataType::kHALF/DataType::kBF16.
    //! \p scale tensor data type defaults to DataType::kFLOAT. For strongly typed networks, it must
    //! have the same data type as the input. The subgraph which terminates with the \p scale tensor
    //! must be a build-time constant. \p outputType output tensor data type, must be
    //! DataType::kINT8 (default), DataType::kFP8 or DataType::kINT4. Future calls to set output
    //! type using setToType or setOutputType must be consistent.
    //!
    //! \return The new quantization layer, or nullptr if it could not be created.
    //!
    IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale, DataType outputType) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addQuantizeV2(input, scale, outputType);
    }

    //!
    //! \brief Add an Einsum layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param equation The equation of the layer
    //! \see IEinsumLayer
    //!
    //! \return The new Einsum layer, or nullptr if it could not be created.
    //!
    IEinsumLayer* addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addEinsum(inputs, nbInputs, equation);
    }

    //!
    //! \brief Add a GridSample layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param grid The grid tensor to the layer.
    //!
    //! \see IGridSampleLayer
    //!
    //! Creates a GridSample layer with a InterpolationMode::kLINEAR, unaligned corners,
    //! and SampleMode::kFILL for 4d-shape input tensors.
    //!
    //! \return The new GridSample layer, or nullptr if it could not be created.
    //!
    IGridSampleLayer* addGridSample(ITensor& input, ITensor& grid) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addGridSample(input, grid);
    }

    //!
    //! \brief Add a non-maximum suppression layer to the network.
    //!
    //! \param boxes The input boxes tensor to the layer.
    //!
    //! \param scores The input scores tensor to the layer.
    //!
    //! \param maxOutputBoxesPerClass The input maxOutputBoxesPerClass tensor to the layer.
    //!
    //! \see INMSLayer
    //!
    //! \return The new NMS layer, or nullptr if it could not be created.
    //!
    INMSLayer* addNMS(ITensor& boxes, ITensor& scores, ITensor& maxOutputBoxesPerClass) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addNMS(boxes, scores, maxOutputBoxesPerClass);
    }

    //!
    //! \brief Add a ReverseSequence layer to the network.
    //!
    //! \param input The input tensor to the layer. Must have rank >= 2.
    //!
    //! \param sequenceLens 1D tensor specifying lengths of sequences to reverse in a batch. The
    //! length of the
    //!        sequenceLens tensor must be equal to the size of the dimension in input tensor
    //!        specified by batchAxis.
    //!
    //! \see IReverseSequenceLayer
    //!
    //! \return The new ReverseSequence layer, or nullptr if it could not be created.
    //!
    IReverseSequenceLayer* addReverseSequence(ITensor& input, ITensor& sequenceLens) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addReverseSequence(input, sequenceLens);
    }

    //!
    //! \brief Add a normalization layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param scale The scale tensor used to scale the normalized output.
    //! \param bias The bias tensor used to scale the normalized output.
    //! \param axesMask The axes on which to perform mean calculations.
    //!        The bit in position i of bitmask axesMask corresponds to explicit dimension i of the
    //!        result. E.g., the least significant bit corresponds to the first explicit dimension
    //!        and the next to least significant bit corresponds to the second explicit dimension.
    //!
    //! The normalization layer works by performing normalization of the tensor \p input on the
    //! specified \p axesMask. The result is then scaled by multiplying with \p scale and adding \p
    //! bias.
    //!
    //! The shape of \p scale and \p bias are expected the be the same, and must have the same rank
    //! and be unidirectionally broadcastable to the shape of \p input.
    //!
    //! \see INormalizationLayer
    //!
    //! \return The new normalization layer, or nullptr if it could not be created.
    //!
    INormalizationLayer*
    addNormalization(ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addNormalization(input, scale, bias, axesMask);
    }

    //!
    //! \brief Return the builder from which this INetworkDefinition was created.
    //!
    //! \see IBuider::createNetworkV2
    //!
    //! \return the builder
    virtual IBuilder& getBuilder() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getBuilder();
    }

    void setProgram(std::shared_ptr<migraphx::program> program)
    {
        program_ = program;
        std::cout << *program_ << std::endl;

        for(auto param : program_->get_main_module()->get_parameters())
        {
            input_tensors_.push_back(std::make_unique<ITensor>(param));
        }

        for(auto param : program_->get_main_module()->get_returns())
        {
            output_tensors_.push_back(std::make_unique<ITensor>(param));
        }
    }

    const migraphx::program* getProgram() const { return program_.get(); }

    protected:
    //     apiv::VNetworkDefinition* mImpl;
    std::shared_ptr<migraphx::program> program_ = std::make_shared<migraphx::program>();
    mutable std::vector<std::unique_ptr<ITensor>> input_tensors_;
    mutable std::vector<std::unique_ptr<ITensor>> output_tensors_;
    mutable std::vector<std::unique_ptr<ILayer>> layers_;
};

//!
//! \brief Represents one or more BuilderFlag values using binary OR
//! operations, e.g., 1U << BuilderFlag::kFP16 | 1U << BuilderFlag::kDEBUG.
//!
//! \see IBuilderConfig::setFlags(), IBuilderConfig::getFlags()
//!
using BuilderFlags = uint32_t;

//!
//! \enum BuilderFlag
//!
//! \brief List of valid modes that the builder can enable when creating an engine from a network
//! definition.
//!
//! \see IBuilderConfig::setFlags(), IBuilderConfig::getFlags()
//!
enum class BuilderFlag : int32_t
{
    //! Enable FP16 layer selection, with FP32 fallback.
    kFP16 = 0,

    //! Enable Int8 layer selection, with FP32 fallback with FP16 fallback if kFP16 also specified.
    kINT8 = 1,

    //! Enable debugging of layers via synchronizing after every layer.
    kDEBUG = 2,

    //! Enable layers marked to execute on GPU if layer cannot execute on DLA.
    kGPU_FALLBACK = 3,

    //! Enable building a refittable engine.
    kREFIT = 4,

    //! Disable reuse of timing information across identical layers.
    kDISABLE_TIMING_CACHE = 5,

    //! Allow (but not require) computations on tensors of type DataType::kFLOAT to use TF32.
    //! TF32 computes inner products by rounding the inputs to 10-bit mantissas before
    //! multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.
    kTF32 = 6,

    //! Allow the builder to examine weights and use optimized functions when weights have suitable
    //! sparsity.
    kSPARSE_WEIGHTS = 7,

    //! Change the allowed parameters in the EngineCapability::kSTANDARD flow to
    //! match the restrictions that EngineCapability::kSAFETY check against for DeviceType::kGPU
    //! and EngineCapability::kDLA_STANDALONE check against the DeviceType::kDLA case. This flag
    //! is forced to true if EngineCapability::kSAFETY at build time if it is unset.
    //!
    //! This flag is only supported in NVIDIA Drive(R) products.
    kSAFETY_SCOPE = 8,

    //! Require that layers execute in specified precisions. Build fails otherwise.
    kOBEY_PRECISION_CONSTRAINTS = 9,

    //! Prefer that layers execute in specified precisions.
    //! Fall back (with warning) to another precision if build would otherwise fail.
    kPREFER_PRECISION_CONSTRAINTS = 10,

    //! Require that no reformats be inserted between a layer and a network I/O tensor
    //! for which ITensor::setAllowedFormats was called.
    //! Build fails if a reformat is required for functional correctness.
    kDIRECT_IO = 11,

    //! Fail if IAlgorithmSelector::selectAlgorithms returns an empty set of algorithms.
    kREJECT_EMPTY_ALGORITHMS = 12,

    //! Restrict to lean runtime operators to provide version forward compatibility
    //! for the plan.
    //!
    //! This flag is only supported by NVIDIA Volta and later GPUs.
    //! This flag is not supported in NVIDIA Drive(R) products.
    kVERSION_COMPATIBLE = 13,

    //! Exclude lean runtime from the plan when version forward compatability is enabled.
    //! By default, this flag is unset, so the lean runtime will be included in the plan.
    //!
    //! If BuilderFlag::kVERSION_COMPATIBLE is not set then the value of this flag will be ignored.
    kEXCLUDE_LEAN_RUNTIME = 14,

    //! Enable FP8 layer selection, with FP32 fallback.
    //!
    //! This flag is not supported with hardware-compatibility mode.
    //!
    //! \see HardwareCompatibilityLevel
    kFP8 = 15,

    //! Emit error when a tactic being timed is not present in the timing cache.
    //! This flag has an effect only when IBuilderConfig has an associated ITimingCache.
    kERROR_ON_TIMING_CACHE_MISS = 16,

    //! Enable DataType::kBF16 layer selection, with FP32 fallback.
    //! This flag is only supported by NVIDIA Ampere and later GPUs.
    kBF16 = 17,

    //! Disable caching of JIT-compilation results during engine build.
    //! By default, JIT-compiled code will be serialized as part of the timing cache, which may
    //! significantly increase the cache size. Setting this flag prevents the code from being
    //! serialized. This flag has an effect only when BuilderFlag::DISABLE_TIMING_CACHE is not set.
    kDISABLE_COMPILATION_CACHE = 18,

    //! Strip the refittable weights from the engine plan file.
    kSTRIP_PLAN = 19,

    //! \deprecated Deprecated in TensorRT 10.0. Superseded by kSTRIP_PLAN.
    kWEIGHTLESS = kSTRIP_PLAN,

    //! Create a refittable engine under the assumption that the refit weights will be identical to
    //! those provided at build time. The resulting engine will have the same performance as a
    //! non-refittable one. All refittable weights can be refitted through the refit API, but if the
    //! refit weights are not identical to the build-time weights, behavior is undefined. When used
    //! alongside 'kSTRIP_PLAN', this flag will result in a small plan file for which weights are
    //! later supplied via refitting. This enables use of a single set of weights with different
    //! inference backends, or with TensorRT plans for multiple GPU architectures.
    kREFIT_IDENTICAL = 20,

    //!
    //! \brief Enable weight streaming for the current engine.
    //!
    //! Weight streaming from the host enables execution of models that do not fit
    //! in GPU memory by allowing TensorRT to intelligently stream network weights
    //! from the CPU DRAM. Please see ICudaEngine::getMinimumWeightStreamingBudget
    //! for the default memory budget when this flag is enabled.
    //!
    //! Enabling this feature changes the behavior of
    //! IRuntime::deserializeCudaEngine to allocate the entire network’s weights
    //! on the CPU DRAM instead of GPU memory. Then,
    //! ICudaEngine::createExecutionContext will determine the optimal split of
    //! weights between the CPU and GPU and place weights accordingly.
    //!
    //! Future TensorRT versions may enable this flag by default.
    //!
    //! \warning Enabling this flag may marginally increase build time.
    //!
    //! \warning Enabling this feature will significantly increase the latency of
    //!          ICudaEngine::createExecutionContext.
    //!
    //! \see IRuntime::deserializeCudaEngine,
    //!      ICudaEngine::getMinimumWeightStreamingBudget,
    //!      ICudaEngine::setWeightStreamingBudget
    //!
    kWEIGHT_STREAMING = 21,
};

//!
//! \enum DeviceType
//! \brief The device that this layer/network will execute on.
//!
//!
enum class DeviceType : int32_t
{
    kGPU = 0, //!< GPU Device
    kDLA = 1, //!< DLA Core
};

//!
//! \brief Represents one or more QuantizationFlag values using binary OR
//! operations.
//!
//! \see IBuilderConfig::getQuantizationFlags(), IBuilderConfig::setQuantizationFlags()
//!
using QuantizationFlags = uint32_t;

//!
//! \enum QuantizationFlag
//!
//! \brief List of valid flags for quantizing the network to int8
//!
//! \see IBuilderConfig::setQuantizationFlag(), IBuilderConfig::getQuantizationFlag()
//!
enum class QuantizationFlag : int32_t
{
    //! Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and
    //! IInt8EntropyCalibrator. The builder always runs the int8 calibration pass before layer
    //! fusion for IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.
    kCALIBRATE_BEFORE_FUSION = 0
};

//!
//! \enum MemoryPoolType
//!
//! \brief The type for memory pools used by TensorRT.
//!
//! \see IBuilderConfig::setMemoryPoolLimit, IBuilderConfig::getMemoryPoolLimit
//!
enum class MemoryPoolType : int32_t
{
    //!
    //! kWORKSPACE is used by TensorRT to store intermediate buffers within an operation.
    //! This defaults to max device memory. Set to a smaller value to restrict tactics that use over
    //! the threshold en masse. For more targeted removal of tactics use the IAlgorithmSelector
    //! interface.
    //!
    kWORKSPACE = 0,

    //!
    //! kDLA_MANAGED_SRAM is a fast software managed RAM used by DLA to communicate within a layer.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 1 MiB.
    //! Orin has capacity of 1 MiB per core.
    //!
    kDLA_MANAGED_SRAM = 1,

    //!
    //! kDLA_LOCAL_DRAM is host RAM used by DLA to share intermediate tensor data across operations.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 1 GiB.
    //!
    kDLA_LOCAL_DRAM = 2,

    //!
    //! kDLA_GLOBAL_DRAM is host RAM used by DLA to store weights and metadata for execution.
    //! The size of this pool must be at least 4 KiB and must be a power of 2.
    //! This defaults to 512 MiB.
    //!
    kDLA_GLOBAL_DRAM = 3,

    //!
    //! kTACTIC_DRAM is the device DRAM used by the optimizer to
    //! run tactics. On embedded devices, where host and device memory are unified, this includes
    //! all host memory required by TensorRT to build the network up to the point of each memory
    //! allocation. This defaults to 75% of totalGlobalMem as reported by cudaGetDeviceProperties
    //! when cudaGetDeviceProperties.embedded is true, and 100% otherwise.
    //!
    kTACTIC_DRAM = 4,

    //!
    //! kTACTIC_SHARED_MEMORY defines the maximum sum of shared memory reserved by the driver and
    //! used for executing CUDA kernels. Adjust this value to restrict tactics that exceed the
    //! specified threshold en masse. The default value is device max capability. This value must
    //! be less than 1GiB.
    //!
    //! The driver reserved shared memory can be queried from cuDeviceGetAttribute(&reservedShmem,
    //! CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK).
    //!
    //! Updating this flag will override the shared memory limit set by \ref
    //! HardwareCompatibilityLevel, which defaults to 48KiB - reservedShmem.
    //!
    kTACTIC_SHARED_MEMORY = 5,
};

//!
//! \enum PreviewFeature
//!
//! \brief Define preview features
//!
//! Preview Features have been fully tested but are not yet as stable as other features in TensorRT.
//! They are provided as opt-in features for at least one release.
//!
enum class PreviewFeature : int32_t
{
    //!
    //! Allows optimization profiles to be shared across execution contexts.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. The default value for this flag is on and can not
    //! be changed.
    //!
    kPROFILE_SHARING_0806 = 0,
};

//!
//! \class IBuilderConfig
//!
//! \brief Holds properties for configuring a builder to produce an engine.
//!
//! \see BuilderFlags
//!
class IBuilderConfig : public INoCopy
{
    public:
    virtual ~IBuilderConfig() noexcept = default;

    //!
    //! \brief Set the number of averaging iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution.
    //! This parameter controls the number of iterations used in averaging.
    //!
    //! \see getAvgTimingIterations()
    //!
    virtual void setAvgTimingIterations(int32_t avgTiming) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setAvgTimingIterations(avgTiming);
    }

    //!
    //! \brief Query the number of averaging iterations.
    //!
    //! By default the number of averaging iterations is 1.
    //!
    //! \see setAvgTimingIterations()
    //!
    int32_t getAvgTimingIterations() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getAvgTimingIterations();
    }

    //!
    //! \brief Configure the builder to target specified EngineCapability flow.
    //!
    //! The flow means a sequence of API calls that allow an application to set up a runtime,
    //! engine, and execution context in order to run inference.
    //!
    //! The supported flows are specified in the EngineCapability enum.
    //!
    void setEngineCapability(EngineCapability capability) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setEngineCapability(capability);
    }

    //!
    //! \brief Query EngineCapability flow configured for the builder.
    //!
    //! By default it returns EngineCapability::kSTANDARD.
    //!
    //! \see setEngineCapability()
    //!
    EngineCapability getEngineCapability() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getEngineCapability();
    }

    //!
    //! \brief Set Int8 Calibration interface.
    //!
    //! The calibrator is to minimize the information loss during the INT8 quantization process.
    //!
    void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setInt8Calibrator(calibrator);
    }

    //!
    //! \brief Get Int8 Calibration interface.
    //!
    IInt8Calibrator* getInt8Calibrator() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getInt8Calibrator();
    }

    //!
    //! \brief Set the build mode flags to turn on builder options for this network.
    //!
    //! The flags are listed in the BuilderFlags enum.
    //! The flags set configuration options to build the network.
    //!
    //! \param builderFlags The build option for an engine.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new
    //! flag.
    //!
    //! \see getFlags()
    //!
    void setFlags(BuilderFlags builderFlags) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setFlags(builderFlags);
    }

    //!
    //! \brief Get the build mode flags for this builder config. Defaults to 0.
    //!
    //! \return The build options as a bitmask.
    //!
    //! \see setFlags()
    //!
    BuilderFlags getFlags() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getFlags();
    }

    //!
    //! \brief clear a single build mode flag.
    //!
    //! clears the builder mode flag from the enabled flags.
    //!
    //! \see setFlags()
    //!
    void clearFlag(BuilderFlag builderFlag) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->clearFlag(builderFlag);
    }

    //!
    //! \brief Set a single build mode flag.
    //!
    //! Add the input builder mode flag to the already enabled flags.
    //!
    //! \see setFlags()
    //!
    void setFlag(BuilderFlag builderFlag) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setFlag(builderFlag);
    }

    //!
    //! \brief Returns true if the build mode flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    bool getFlag(BuilderFlag builderFlag) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getFlag(builderFlag);
    }

    //!
    //! \brief Set the device that this layer must execute on.
    //!
    //! \param layer which layer to execute.
    //! \param deviceType that this layer must execute on.
    //! If DeviceType is not set or is reset, TensorRT will use the default DeviceType set in the
    //! builder.
    //!
    //! \note The device type for a layer must be compatible with the safety flow (if specified).
    //! For example a layer cannot be marked for DLA execution while the builder is configured for
    //! kSAFETY.
    //!
    //! \see getDeviceType()
    //!
    void setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDeviceType(layer, deviceType);
    }

    //!
    //! \brief Get the device that this layer executes on.
    //!
    //! \return Returns DeviceType of the layer.
    //!
    DeviceType getDeviceType(ILayer const* layer) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDeviceType(layer);
    }

    //!
    //! \brief whether the DeviceType has been explicitly set for this layer
    //!
    //! \return true if device type is not default
    //!
    //! \see setDeviceType() getDeviceType() resetDeviceType()
    //!
    bool isDeviceTypeSet(ILayer const* layer) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isDeviceTypeSet(layer);
    }

    //!
    //! \brief reset the DeviceType for this layer
    //!
    //! \see setDeviceType() getDeviceType() isDeviceTypeSet()
    //!
    void resetDeviceType(ILayer const* layer) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->resetDeviceType(layer);
    }

    //!
    //! \brief Checks if a layer can run on DLA.
    //!
    //! \return status true if the layer can on DLA else returns false.
    //!
    bool canRunOnDLA(ILayer const* layer) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->canRunOnDLA(layer);
    }

    //!
    //! \brief Sets the DLA core used by the network. Defaults to -1.
    //!
    //! \param dlaCore The DLA core to execute the engine on, in the range [0,getNbDlaCores()).
    //!
    //! This function is used to specify which DLA core to use via indexing, if multiple DLA cores
    //! are available.
    //!
    //! \warning if getNbDLACores() returns 0, then this function does nothing.
    //!
    //! \see IRuntime::setDLACore() getDLACore()
    //!
    void setDLACore(int32_t dlaCore) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDLACore(dlaCore);
    }

    //!
    //! \brief Get the DLA core that the engine executes on.
    //!
    //! \return assigned DLA core or -1 for DLA not present or unset.
    //!
    int32_t getDLACore() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDLACore();
    }

    //!
    //! \brief Sets the default DeviceType to be used by the builder. It ensures that all the layers
    //! that can run on this device will run on it, unless setDeviceType is used to override the
    //! default DeviceType for a layer.
    //!
    //! \see getDefaultDeviceType()
    //!
    void setDefaultDeviceType(DeviceType deviceType) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setDefaultDeviceType(deviceType);
    }

    //!
    //! \brief Get the default DeviceType which was set by setDefaultDeviceType.
    //!
    //! By default it returns DeviceType::kGPU.
    //!
    DeviceType getDefaultDeviceType() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getDefaultDeviceType();
    }

    //!
    //! \brief Resets the builder configuration to defaults.
    //!
    //! Useful for initializing a builder config object to its original state.
    //!
    void reset() noexcept
    {
        pass("Not Implemented", true);
        // mImpl->reset();
    }

    //!
    //! \brief Set the cuda stream that is used to profile this network.
    //!
    //! \param stream The cuda stream used for profiling by the builder.
    //!
    //! \see getProfileStream()
    //!
    void setProfileStream(const hipStream_t stream) noexcept
    {
        pass("Not Implemented", false);
        // return mImpl->setProfileStream(stream);
    }

    //!
    //! \brief Get the cuda stream that is used to profile this network.
    //!
    //! \return The cuda stream set by setProfileStream, nullptr if setProfileStream has not been
    //! called.
    //!
    //! \see setProfileStream()
    //!
    hipStream_t getProfileStream() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProfileStream();
    }

    //!
    //! \brief Add an optimization profile.
    //!
    //! This function must be called at least once if the network has dynamic or shape input
    //! tensors. This function may be called at most once when building a refittable engine, as more
    //! than a single optimization profile are not supported for refittable engines.
    //!
    //! \param profile The new optimization profile, which must satisfy profile->isValid() == true
    //!
    //! \return The index of the optimization profile (starting from 0) if the input is valid, or -1
    //! if the input is
    //!         not valid.
    //!
    int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->addOptimizationProfile(profile);
    }

    //!
    //! \brief Get number of optimization profiles.
    //!
    //! This is one higher than the index of the last optimization profile that has be defined (or
    //! zero, if none has been defined yet).
    //!
    //! \return The number of the optimization profiles.
    //!
    int32_t getNbOptimizationProfiles() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbOptimizationProfiles();
    }

    //!
    //! \brief Set verbosity level of layer information exposed in NVTX annotations and
    //! IEngineInspector.
    //!
    //! Control how much layer information will be exposed in NVTX annotations and IEngineInspector.
    //!
    //! \see ProfilingVerbosity, getProfilingVerbosity(), IEngineInspector
    //!
    void setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setProfilingVerbosity(verbosity);
    }

    //!
    //! \brief Get verbosity level of layer information exposed in NVTX annotations and
    //! IEngineInspector.
    //!
    //! Get the current setting of verbosity level of layer information exposed in
    //! NVTX annotations and IEngineInspector. Default value is
    //! ProfilingVerbosity::kLAYER_NAMES_ONLY.
    //!
    //! \see ProfilingVerbosity, setProfilingVerbosity(), IEngineInspector
    //!
    ProfilingVerbosity getProfilingVerbosity() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProfilingVerbosity();
    }

    //!
    //! \brief Set Algorithm Selector.
    //!
    //! \param selector The algorithm selector to be set in the build config.
    void setAlgorithmSelector(IAlgorithmSelector* selector) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setAlgorithmSelector(selector);
    }

    //!
    //! \brief Get Algorithm Selector.
    //!
    IAlgorithmSelector* getAlgorithmSelector() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getAlgorithmSelector();
    }

    //!
    //! \brief Add a calibration profile.
    //!
    //! Calibration optimization profile must be set if int8 calibration is used to set scales for a
    //! network with runtime dimensions.
    //!
    //! \param profile The new calibration profile, which must satisfy profile->isValid() == true or
    //! be nullptr. MIN and MAX values will be overwritten by kOPT.
    //!
    //! \return True if the calibration profile was set correctly.
    //!
    bool setCalibrationProfile(IOptimizationProfile const* profile) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setCalibrationProfile(profile);
    }

    //!
    //! \brief Get the current calibration profile.
    //!
    //! \return A pointer to the current calibration profile or nullptr if calibration profile is
    //! unset.
    //!
    IOptimizationProfile const* getCalibrationProfile() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getCalibrationProfile();
    }

    //!
    //! \brief Set the quantization flags.
    //!
    //! The flags are listed in the QuantizationFlag enum.
    //! The flags set configuration options to quantize the network in int8.
    //!
    //! \param flags The quantization flags.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new
    //! flag.
    //!
    //! \see getQuantizationFlags()
    //!
    void setQuantizationFlags(QuantizationFlags flags) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setQuantizationFlags(flags);
    }

    //!
    //! \brief Get the quantization flags.
    //!
    //! \return The quantization flags as a bitmask.
    //!
    //! \see setQuantizationFlag()
    //!
    QuantizationFlags getQuantizationFlags() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getQuantizationFlags();
    }

    //!
    //! \brief clear a quantization flag.
    //!
    //! Clears the quantization flag from the enabled quantization flags.
    //!
    //! \see setQuantizationFlags()
    //!
    void clearQuantizationFlag(QuantizationFlag flag) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->clearQuantizationFlag(flag);
    }

    //!
    //! \brief Set a single quantization flag.
    //!
    //! Add the input quantization flag to the already enabled quantization flags.
    //!
    //! \see setQuantizationFlags()
    //!
    void setQuantizationFlag(QuantizationFlag flag) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setQuantizationFlag(flag);
    }

    //!
    //! \brief Returns true if the quantization flag is set.
    //!
    //! \see getQuantizationFlags()
    //!
    //! \return True if quantization flag is set, false if unset.
    //!
    bool getQuantizationFlag(QuantizationFlag flag) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getQuantizationFlag(flag);
    }

    //!
    //! \brief Set tactic sources.
    //!
    //! This bitset controls which tactic sources TensorRT is allowed to use for tactic
    //! selection.
    //!
    //! Multiple tactic sources may be combined with a bitwise OR operation. For example,
    //! to enable cublas and cublasLt as tactic sources, use a value of:
    //!
    //! 1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U <<
    //! static_cast<uint32_t>(TacticSource::kCUBLAS_LT)
    //!
    //! \see getTacticSources
    //!
    //! \return true if the tactic sources in the build configuration were updated.
    //!         The tactic sources in the build configuration will not be updated if the provided
    //!         value is invalid.
    //!
    bool setTacticSources(TacticSources tacticSources) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setTacticSources(tacticSources);
    }

    //!
    //! \brief Get tactic sources.
    //!
    //! Get the tactic sources currently set in the engine build
    //! configuration.
    //!
    //! \see setTacticSources()
    //!
    //! \return tactic sources
    //!
    TacticSources getTacticSources() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTacticSources();
    }

    //!
    //! \brief Create timing cache
    //!
    //! Create ITimingCache instance from serialized raw data. The created timing cache doesn’t
    //! belong to a specific IBuilderConfig. It can be shared by multiple builder instances. Call
    //! setTimingCache() before launching a builder to attach cache to builder instance.
    //!
    //! \param blob A pointer to the raw data that contains serialized timing cache
    //! \param size The size in bytes of the serialized timing cache. Size 0 means create a new
    //! cache from scratch
    //!
    //! \see setTimingCache
    //!
    //! \return the pointer to ITimingCache created
    //!
    ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->createTimingCache(blob, size);
    }

    //!
    //! \brief Attach a timing cache to IBuilderConfig
    //!
    //! The timing cache has verification header to make sure the provided cache can be used in
    //! current environment. A failure will be reported if the CUDA device property in the provided
    //! cache is different from current environment. ignoreMismatch = true skips strict verification
    //! and allows loading cache created from a different device.
    //!
    //! The cache must not be destroyed until after the engine is built.
    //!
    //! \param cache the timing cache to be used
    //! \param ignoreMismatch whether or not allow using a cache that contains different CUDA device
    //! property
    //!
    //! \return true if set successfully, false otherwise
    //!
    //! \warning Using cache generated from devices with different CUDA device properties may lead
    //! to
    //!          functional/performance bugs.
    //!
    bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setTimingCache(cache, ignoreMismatch);
    }

    //!
    //! \brief Get the pointer to the timing cache from current IBuilderConfig
    //!
    //! \return pointer to the timing cache used in current IBuilderConfig
    //!
    ITimingCache const* getTimingCache() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getTimingCache();
    }

    //!
    //! \brief Set the memory size for the memory pool.
    //!
    //! TensorRT layers access different memory pools depending on the operation.
    //! This function sets in the IBuilderConfig the size limit, specified by \p poolSize,
    //! for the corresponding memory pool, specified by \p pool.
    //! TensorRT will build a plan file that is constrained by these limits or report
    //! which constraint caused the failure.
    //!
    //! If the size of the pool, specified by \p poolSize, fails to meet the size requirements
    //! for the pool, this function does nothing and emits the recoverable error,
    //! ErrorCode::kINVALID_ARGUMENT, to the registered IErrorRecorder.
    //!
    //! If the size of the pool is larger than the maximum possible value for the
    //! configuration, this function does nothing and emits ErrorCode::kUNSUPPORTED_STATE.
    //!
    //! If the pool does not exist on the requested device type when building
    //! the network, a warning is emitted to the logger, and the memory pool
    //! value is ignored.
    //!
    //! Refer to MemoryPoolType to see the size requirements for each pool.
    //!
    //! \param pool The memory pool to limit the available memory for.
    //! \param poolSize The size of the pool in bytes.
    //!
    //! \see getMemoryPoolLimit, MemoryPoolType
    //!
    void setMemoryPoolLimit(MemoryPoolType pool, std::size_t poolSize) noexcept
    {
        // TODO log that setMemoryPoolLimit is a noop, pop up a warning, or do both
        // mImpl->setMemoryPoolLimit(pool, poolSize);
    }

    //!
    //! \brief Get the memory size limit of the memory pool.
    //!
    //! Retrieve the memory size limit of the corresponding pool in bytes.
    //! If setMemoryPoolLimit for the pool has not been called, this returns the default
    //! value used by TensorRT. This default value is not necessarily the maximum possible
    //! value for that configuration.
    //!
    //! \param pool The memory pool to get the limit for.
    //!
    //! \returns The size of the memory limit, in bytes, for the corresponding pool.
    //!
    //! \see setMemoryPoolLimit
    //!
    std::size_t getMemoryPoolLimit(MemoryPoolType pool) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMemoryPoolLimit(pool);
    }

    //!
    //! \brief Enable or disable a specific preview feature
    //!
    //! Allows enabling or disabling experimental features, which are not enabled by default in the
    //! current release.
    //!
    //! Refer to PreviewFeature for additional information, and a list of the available features.
    //!
    //! \param feature the feature to enable / disable
    //! \param enable true for enable, false for disable
    //!
    //! \see PreviewFeature, getPreviewFeature
    //!
    void setPreviewFeature(PreviewFeature feature, bool enable) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setPreviewFeature(feature, enable);
    }

    //!
    //! \brief Get status of preview feature
    //!
    //! \param feature the feature to query
    //!
    //! \returns true if the \p feature is enabled, false otherwise
    //!
    //! \see PreviewFeature, setPreviewFeature
    //!
    bool getPreviewFeature(PreviewFeature feature) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getPreviewFeature(feature);
    }

    //!
    //! \brief Set builder optimization level
    //!
    //! Set the builder optimization level. Setting a higher optimization
    //! level allows the optimizer to spend more time searching for optimization opportunities. The
    //! resulting engine may have better performance compared to an engine built with a lower
    //! optimization level.
    //!
    //! The default optimization level is 3. Valid values include integers from 0 to the maximum
    //! optimization level, which is currently 5. Setting it to greater than the maximum level
    //! results in behavior identical to the maximum level.
    //!
    //! Below are the descriptions about each builder optimization level:
    //!
    //! - Level 0: This enables the fastest compilation by disabling dynamic kernel generation and
    //! selecting the first
    //!   tactic that succeeds in execution. This will also not respect a timing cache.
    //! - Level 1: Available tactics are sorted by heuristics, but only the top are tested to select
    //! the best. If a
    //!   dynamic kernel is generated its compile optimization is low.
    //! - Level 2: Available tactics are sorted by heuristics, but only the fastest tactics are
    //! tested to select the
    //!   best.
    //! - Level 3: Apply heuristics to see if a static precompiled kernel is applicable or if a new
    //! one has to be
    //!   compiled dynamically.
    //! - Level 4: Always compiles a dynamic kernel.
    //! - Level 5: Always compiles a dynamic kernel and compares it to static kernels.
    //!
    //! \param level The optimization level to set to. Must be non-negative.
    //!
    //! \see getBuilderOptimizationLevel
    //!
    void setBuilderOptimizationLevel(int32_t level) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setBuilderOptimizationLevel(level);
    }

    //!
    //! \brief Get builder optimization level
    //!
    //! \returns the current builder optimization level
    //!
    //! \see setBuilderOptimizationLevel
    //!
    int32_t getBuilderOptimizationLevel() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getBuilderOptimizationLevel();
    }

    //!
    //! \brief Set the hardware compatibility level.
    //!
    //! Hardware compatibility allows an engine to run on GPU
    //! architectures other than that of the GPU where the engine was
    //! built.
    //!
    //! The default hardware compatibility level is HardwareCompatibilityLevel::kNONE.
    //!
    //! \param hardwareCompatibilityLevel The level of hardware
    //!        compatibility.
    //!
    void
    setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setHardwareCompatibilityLevel(hardwareCompatibilityLevel);
    }

    //!
    //! \brief Get the hardware compatibility level.
    //!
    //! \return hardwareCompatibilityLevel The level of hardware
    //!        compatibility.
    //!
    //! \see setHardwareCompatiblityLevel()
    //!
    HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getHardwareCompatibilityLevel();
    }

    //!
    //! \brief Set the plugin libraries to be serialized with version-compatible engines.
    //!
    //! Each entry in the list of libraries must be unique.
    //!
    //! \param paths The paths of plugin libraries.
    //! \param nbPaths The number of paths.
    //!
    void setPluginsToSerialize(char const* const* paths, int32_t nbPaths) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setPluginsToSerialize(paths, nbPaths);
    }

    //!
    //! \brief Get the plugin library path to be serialized with version-compatible engines.
    //!
    //! \param index Index of the plugin library path in the list.  Should be in the range `[0,
    //! getNbPluginsToSerialize())`.
    //!
    //! \return The path to the plugin library.
    //!
    char const* getPluginToSerialize(int32_t index) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getPluginToSerialize(index);
    }

    //!
    //! \brief Get the number of plugin library paths to be serialized with version-compatible
    //! engines.
    //!
    //! \return The number of paths.
    //!
    int32_t getNbPluginsToSerialize() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbPluginsToSerialize();
    }

    //!
    //! \brief Set the maximum number of auxiliary streams that TRT is allowed to use.
    //!
    //! If the network contains operators that can run in parallel, TRT can execute them using
    //! auxiliary streams in addition to the one provided to the IExecutionContext::enqueueV3()
    //! call.
    //!
    //! The default maximum number of auxiliary streams is determined by the heuristics in TensorRT
    //! on whether enabling multi-stream would improve the performance. This behavior can be
    //! overridden by calling this API to set the maximum number of auxiliary streams explicitly.
    //! Set this to 0 to enforce single-stream inference.
    //!
    //! The resulting engine may use fewer auxiliary streams than the maximum if the network does
    //! not contain enough parallelism or if TensorRT determines that using more auxiliary streams
    //! does not help improve the performance.
    //!
    //! \note Allowing more auxiliary streams does not always give better performance since there
    //! will be synchronizations overhead between streams. Using CUDA graphs at runtime can help
    //! reduce the overhead caused by cross-stream synchronizations.
    //!
    //! \note Using more auxiliary leads to more memory usage at runtime since some activation
    //! memory blocks will not be able to be reused.
    //!
    //! \param nbStreams The maximum number of auxiliary streams that TRT is allowed to use.
    //!
    //! \see getMaxAuxStreams(), ICudaEngine::getNbAuxStreams(), IExecutionContext::setAuxStreams()
    //!
    void setMaxAuxStreams(int32_t nbStreams) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setMaxAuxStreams(nbStreams);
    }

    //!
    //! \brief Get the maximum number of auxiliary streams that TRT is allowed to use.
    //!
    //! \see setMaxAuxStreams()
    //!
    int32_t getMaxAuxStreams() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMaxAuxStreams();
    }

    //!
    //! \brief Sets the progress monitor for building a network.
    //!
    //! \param monitor The progress monitor to assign to the IBuilderConfig.
    //!
    //! The progress monitor signals to the application when different phases of
    //! the compiler are being executed. Setting to nullptr unsets the monitor so
    //! that the application is not signaled.
    //!
    //! \see IBuilderConfig::getProgressMonitor
    //!
    void setProgressMonitor(IProgressMonitor* monitor) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setProgressMonitor(monitor);
    }

    //!
    //! \return The progress monitor set by the application or nullptr.
    //!
    //! \see IBuilderConfig::setProgressMonitor
    //!
    IProgressMonitor* getProgressMonitor() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getProgressMonitor();
    }

    // protected:
    //     apiv::VBuilderConfig* mImpl;
};

//!
//! \class IBuilder
//!
//! \brief Builds an engine from a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IBuilder : public INoCopy
{
    public:
    virtual ~IBuilder() noexcept = default;

    //!
    //! \brief Determine whether the platform has fast native fp16.
    //!
    bool platformHasFastFp16() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->platformHasFastFp16();
    }

    //!
    //! \brief Determine whether the platform has fast native int8.
    //!
    bool platformHasFastInt8() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->platformHasFastInt8();
    }

    //!
    //! \brief Get the maximum batch size DLA can support.
    //! For any tensor the total volume of index dimensions combined(dimensions other than CHW) with
    //! the requested batch size should not exceed the value returned by this function.
    //!
    //! \warning getMaxDLABatchSize does not work with dynamic shapes.
    //!
    int32_t getMaxDLABatchSize() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMaxDLABatchSize();
    }

    //!
    //! \brief Return the number of DLA engines available to this builder.
    //!
    int32_t getNbDLACores() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getNbDLACores();
    }

    //!
    //! \brief Set the GPU allocator.
    //!
    //! \param allocator Set the GPU allocator to be used by the builder. All GPU memory acquired
    //! will use this allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! \note This allocator will be passed to any engines created via the builder; thus the
    //! lifetime of the allocator must span the lifetime of those engines as well as that of the
    //! builder. If nullptr is passed, the default allocator will be used.
    //!
    void setGpuAllocator(IGpuAllocator* allocator) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setGpuAllocator(allocator);
    }

    //!
    //! \brief Create a builder configuration object.
    //!
    //! \see IBuilderConfig
    //!
    IBuilderConfig* createBuilderConfig() noexcept
    {
        return new IBuilderConfig{};
        // return mImpl->createBuilderConfig();
    }

    //!
    //! \brief Create a network definition object
    //!
    //! Creates a network definition object with immutable properties specified using the flags
    //! parameter.
    //!
    //! createNetworkV2 supports creating network with properties from
    //! NetworkDefinitionCreationFlags.
    //!
    //! CreateNetworkV2 supports dynamic shapes and explicit batch dimensions by default.
    //!
    //! createNetworkV2 with NetworkDefinitionCreationFlag::kSTRONGLY_TYPED flag supports creating a
    //! strongly typed plan where tensor data types are inferred from network input types and
    //! operator type specification.
    //!
    //! \param flags Bitset of NetworkDefinitionCreationFlags specifying network properties combined
    //! with bitwise OR.
    //!             e.g., 1U << NetworkDefinitionCreationFlag::kSTRONGLY_TYPED
    //!
    //! \see INetworkDefinition, NetworkDefinitionCreationFlags
    //!
    INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept
    {
        return new INetworkDefinition{};
        // return mImpl->createNetworkV2(flags);
    }

    //!
    //! \brief Create a new optimization profile.
    //!
    //! If the network has any dynamic input tensors, the appropriate calls to setDimensions() must
    //! be made. Likewise, if there are any shape input tensors, the appropriate calls to
    //! setShapeValues() are required. The builder retains ownership of the created optimization
    //! profile and returns a raw pointer, i.e. the users must not attempt to delete the returned
    //! pointer.
    //!
    //! \see IOptimizationProfile
    //!
    IOptimizationProfile* createOptimizationProfile() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->createOptimizationProfile();
    }

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during
    //! execution. This function will call incRefCount of the registered ErrorRecorder at least
    //! once. Setting recorder to nullptr unregisters the recorder with the interface, resulting in
    //! a call to decRefCount if a recorder has been registered.
    //!
    //! If an error recorder is not set, messages will be sent to the global log stream.
    //!
    //! \param recorder The error recorder to register with this interface.
    //!
    //! \see getErrorRecorder()
    //!
    void setErrorRecorder(IErrorRecorder* recorder) noexcept
    {
        pass("Not Implemented", true);
        // mImpl->setErrorRecorder(recorder);
    }

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class.
    //! A nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    IErrorRecorder* getErrorRecorder() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getErrorRecorder();
    }

    //!
    //! \brief Resets the builder state to default values.
    //!
    void reset() noexcept
    {
        pass("Not Implemented", true);
        // mImpl->reset();
    }

    //!
    //! \brief Determine whether the platform has TF32 support.
    //!
    bool platformHasTf32() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->platformHasTf32();
    }

    //!
    //! \brief Builds and serializes a network for the given INetworkDefinition and IBuilderConfig.
    //!
    //! This function allows building and serialization of a network without creating an engine.
    //!
    //! \param network Network definition.
    //! \param config Builder configuration.
    //!
    //! \return A pointer to a IHostMemory object that contains a serialized network.
    //!
    //! \note This function will synchronize the cuda stream returned by \p
    //! config.getProfileStream() before returning.
    //!
    //! \see INetworkDefinition, IBuilderConfig, IHostMemory
    //!
    IHostMemory* buildSerializedNetwork(INetworkDefinition& network,
                                        IBuilderConfig& config) noexcept
    {
        migraphx::program p = *network.getProgram();
        try
        {
            std::cout << p << std::endl;
            p.compile(migraphx::make_target("gpu"));
            std::cout << p << std::endl;
        }
        catch(migraphx::exception& e)
        {
            // TODO write to error recorder/logger
            return nullptr;
        }
        serialized_networks_.push_back(migraphx::save_buffer(p));
        auto&& current_network = serialized_networks_.back();
        return new IHostMemory{reinterpret_cast<void*>(current_network.data()),
                               current_network.size(),
                               DataType::kUINT8};
    }

    //!
    //! \brief Checks that a network is within the scope of the IBuilderConfig settings.
    //!
    //! \param network The network definition to check for configuration compliance.
    //! \param config The configuration of the builder to use when checking \p network.
    //!
    //! Given an INetworkDefinition, \p network, and an IBuilderConfig, \p config, check if
    //! the network falls within the constraints of the builder configuration based on the
    //! EngineCapability, BuilderFlag, and DeviceType. If the network is within the constraints,
    //! then the function returns true, and false if a violation occurs. This function reports
    //! the conditions that are violated to the registered ErrorRecorder.
    //!
    //! \return True if network is within the scope of the restrictions specified by the builder
    //! config, false otherwise.
    //!
    //! \note This function will synchronize the cuda stream returned by \p
    //! config.getProfileStream() before returning.
    //!
    bool isNetworkSupported(INetworkDefinition const& network,
                            IBuilderConfig const& config) const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->isNetworkSupported(network, config);
    }

    //!
    //! \brief get the logger with which the builder was created
    //!
    //! \return the logger
    //!
    ILogger* getLogger() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getLogger();
    }

    //!
    //! \brief Set the maximum number of threads.
    //!
    //! \param maxThreads The maximum number of threads that can be used by the builder.
    //!
    //! \return True if successful, false otherwise.
    //!
    //! The default value is 1 and includes the current thread.
    //! A value greater than 1 permits TensorRT to use multi-threaded algorithms.
    //! A value less than 1 triggers a kINVALID_ARGUMENT error.
    //!
    bool setMaxThreads(int32_t maxThreads) noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->setMaxThreads(maxThreads);
    }

    //!
    //! \brief get the maximum number of threads that can be used by the builder.
    //!
    //! Retrieves the maximum number of threads that can be used by the builder.
    //!
    //! \return The maximum number of threads that can be used by the builder.
    //!
    //! \see setMaxThreads()
    //!
    int32_t getMaxThreads() const noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getMaxThreads();
    }

    //!
    //! \brief get the local plugin registry that can be used by the builder.
    //!
    //! \return The local plugin registry that can be used by the builder.
    //!
    IPluginRegistry& getPluginRegistry() noexcept
    {
        pass("Not Implemented", true);
        // return mImpl->getPluginRegistry();
    }

    // protected:
    //     apiv::VBuilder* mImpl;

    private:
    // TODO The TRT builder has some sort of reference counting for memory objects it has allocated.
    // If it gets destroyed before the IHostMemory objects it passes outside its scope, a warning
    // gets generated.
    // Also look into the probability that it maintains its own memory pools that can have its size
    // set by another apy call.
    std::vector<std::vector<char>> serialized_networks_;
};

//!
//! \brief Create an instance of an IBuilder class.
//!
//! \param logger The logging class for the builder.
//!
//! unnamed namespace avoids linkage surprises when linking objects built with different versions of
//! this header.
//!
inline IBuilder* createInferBuilder(ILogger& logger) noexcept { return new IBuilder{}; }

namespace consistency {

//!
//! \class IConsistencyChecker
//!
//! \brief Validates a serialized engine blob.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API
//! and ABI.
//!
class IConsistencyChecker
{
    public:
    //!
    //! \brief Check that a blob that was input to createConsistencyChecker method represents a
    //! valid engine.
    //!
    //! \return true if the original blob encoded an engine that belongs to valid engine domain with
    //! target capability EngineCapability::kSAFETY, false otherwise.
    //!
    bool validate() const noexcept
    {
        pass("Not Implemented", false);
        return true;
    }

    //!
    //! \brief De-allocates any internally allocated memory.
    //!
    virtual ~IConsistencyChecker() = default;

    protected:
    // apiv::VConsistencyChecker* mImpl;
    IConsistencyChecker()                                            = default;
    IConsistencyChecker(IConsistencyChecker const& other)            = delete;
    IConsistencyChecker& operator=(IConsistencyChecker const& other) = delete;
    IConsistencyChecker(IConsistencyChecker&& other)                 = delete;
    IConsistencyChecker& operator=(IConsistencyChecker&& other)      = delete;
};
} // namespace consistency

} // namespace mgxinfer1

//!
//! \typedef SubGraph_t
//!
//! \brief The data structure containing the parsing capability of
//! a set of nodes in an ONNX graph.
//!
typedef std::pair<std::vector<size_t>, bool> SubGraph_t;

//!
//! \typedef SubGraphCollection_t
//!
//! \brief The data structure containing all SubGraph_t partitioned
//! out of an ONNX graph.
//!
typedef std::vector<SubGraph_t> SubGraphCollection_t;

namespace mgxonnxparser {

//!
//! \enum ErrorCode
//!
//! \brief The type of error that the parser or refitter may return
//!
enum class ErrorCode : int
{
    kSUCCESS                   = 0,
    kINTERNAL_ERROR            = 1,
    kMEM_ALLOC_FAILED          = 2,
    kMODEL_DESERIALIZE_FAILED  = 3,
    kINVALID_VALUE             = 4,
    kINVALID_GRAPH             = 5,
    kINVALID_NODE              = 6,
    kUNSUPPORTED_GRAPH         = 7,
    kUNSUPPORTED_NODE          = 8,
    kUNSUPPORTED_NODE_ATTR     = 9,
    kUNSUPPORTED_NODE_INPUT    = 10,
    kUNSUPPORTED_NODE_DATATYPE = 11,
    kUNSUPPORTED_NODE_DYNAMIC  = 12,
    kUNSUPPORTED_NODE_SHAPE    = 13,
    kREFIT_FAILED              = 14
};

//!
//! \brief Represents one or more OnnxParserFlag values using binary OR
//! operations, e.g., 1U << OnnxParserFlag::kNATIVE_INSTANCENORM
//!
//! \see IParser::setFlags() and IParser::getFlags()
//!
using OnnxParserFlags = uint32_t;

enum class OnnxParserFlag : int32_t
{
    //! Parse the ONNX model into the INetworkDefinition with the intention of using TensorRT's
    //! native layer implementation over the plugin implementation for InstanceNormalization nodes.
    //! This flag is required when building version-compatible or hardware-compatible engines.
    //! This flag is set to be ON by default.
    kNATIVE_INSTANCENORM = 0
};

//!
//! \class IParserError
//!
//! \brief an object containing information about an error
//!
class IParserError
{
    public:
    //!
    //!\brief the error code.
    //!
    virtual ErrorCode code() const = 0;
    //!
    //!\brief description of the error.
    //!
    virtual char const* desc() const = 0;
    //!
    //!\brief source file in which the error occurred.
    //!
    virtual char const* file() const = 0;
    //!
    //!\brief source line at which the error occurred.
    //!
    virtual int line() const = 0;
    //!
    //!\brief source function in which the error occurred.
    //!
    virtual char const* func() const = 0;
    //!
    //!\brief index of the ONNX model node in which the error occurred.
    //!
    virtual int node() const = 0;
    //!
    //!\brief name of the node in which the error occurred.
    //!
    virtual char const* nodeName() const = 0;
    //!
    //!\brief name of the node operation in which the error occurred.
    //!
    virtual char const* nodeOperator() const = 0;
    //!
    //!\brief A list of the local function names, from the top level down, constituting the current
    //!             stack trace in which the error occurred. A top-level node that is not inside any
    //!             local function would return a nullptr.
    //!
    virtual char const* const* localFunctionStack() const = 0;
    //!
    //!\brief The size of the stack of local functions at the point where the error occurred.
    //!             A top-level node that is not inside any local function would correspond to
    //              a stack size of 0.
    //!
    virtual int32_t localFunctionStackSize() const = 0;

    protected:
    virtual ~IParserError() {}
};

//!
//! \class IParser
//!
//! \brief an object for parsing ONNX models into a TensorRT network definition
//!
class IParser
{
    public:
    //!
    //! \brief Parse a serialized ONNX model into the TensorRT network.
    //!         This method has very limited diagnostics. If parsing the serialized model
    //!         fails for any reason (e.g. unsupported IR version, unsupported opset, etc.)
    //!         it the user responsibility to intercept and report the error.
    //!         To obtain a better diagnostic, use the parseFromFile method below.
    //!
    //! \param serialized_onnx_model Pointer to the serialized ONNX model
    //! \param serialized_onnx_model_size Size of the serialized ONNX model
    //!        in bytes
    //! \param model_path Absolute path to the model file for loading external weights if required
    //! \return true if the model was parsed successfully
    //! \see getNbErrors() getError()
    //!
    virtual bool parse(void const* serialized_onnx_model,
                       size_t serialized_onnx_model_size,
                       const char* model_path = nullptr) = 0;

    //!
    //! \brief Parse an onnx model file, which can be a binary protobuf or a text onnx model
    //!         calls parse method inside.
    //!
    //! \param onnxModelFile name
    //! \param verbosity Level
    //!
    //! \return true if the model was parsed successfully
    //!
    //!
    virtual bool parseFromFile(const char* onnxModelFile, int verbosity) = 0;

    //!
    //!\brief Check whether TensorRT supports a particular ONNX model.
    //! 	       If the function returns True, one can proceed to engine building
    //! 	       without having to call \p parse or \p parseFromFile.
    //!
    //! \param serialized_onnx_model Pointer to the serialized ONNX model
    //! \param serialized_onnx_model_size Size of the serialized ONNX model
    //!        in bytes
    //! \param sub_graph_collection Container to hold supported subgraphs
    //! \param model_path Absolute path to the model file for loading external weights if required
    //! \return true if the model is supported
    //!
    virtual bool supportsModel(void const* serialized_onnx_model,
                               size_t serialized_onnx_model_size,
                               SubGraphCollection_t& sub_graph_collection,
                               const char* model_path = nullptr) = 0;

    //!
    //!\brief Parse a serialized ONNX model into the TensorRT network
    //! with consideration of user provided weights
    //!
    //! \param serialized_onnx_model Pointer to the serialized ONNX model
    //! \param serialized_onnx_model_size Size of the serialized ONNX model
    //!        in bytes
    //! \return true if the model was parsed successfully
    //! \see getNbErrors() getError()
    //!
    virtual bool parseWithWeightDescriptors(void const* serialized_onnx_model,
                                            size_t serialized_onnx_model_size) = 0;

    //!
    //!\brief Returns whether the specified operator may be supported by the
    //!         parser.
    //!
    //! Note that a result of true does not guarantee that the operator will be
    //! supported in all cases (i.e., this function may return false-positives).
    //!
    //! \param op_name The name of the ONNX operator to check for support
    //!
    virtual bool supportsOperator(const char* op_name) const = 0;

    //!
    //!\brief Get the number of errors that occurred during prior calls to
    //!         \p parse
    //!
    //! \see getError() clearErrors() IParserError
    //!
    virtual int getNbErrors() const = 0;

    //!
    //!\brief Get an error that occurred during prior calls to \p parse
    //!
    //! \see getNbErrors() clearErrors() IParserError
    //!
    virtual IParserError const* getError(int index) const = 0;

    //!
    //!\brief Clear errors from prior calls to \p parse
    //!
    //! \see getNbErrors() getError() IParserError
    //!
    virtual void clearErrors() = 0;

    virtual ~IParser() noexcept = default;

    //!
    //! \brief Query the plugin libraries needed to implement operations used by the parser in a
    //! version-compatible engine.
    //!
    //! This provides a list of plugin libraries on the filesystem needed to implement operations
    //! in the parsed network.  If you are building a version-compatible engine using this network,
    //! provide this list to IBuilderConfig::setPluginsToSerialize to serialize these plugins along
    //! with the version-compatible engine, or, if you want to ship these plugin libraries
    //! externally to the engine, ensure that IPluginRegistry::loadLibrary is used to load these
    //! libraries in the appropriate runtime before deserializing the corresponding engine.
    //!
    //! \param[out] nbPluginLibs Returns the number of plugin libraries in the array, or -1 if there
    //! was an error. \return Array of `nbPluginLibs` C-strings describing plugin library paths on
    //! the filesystem if nbPluginLibs > 0, or nullptr otherwise.  This array is owned by the
    //! IParser, and the pointers in the array are only valid until the next call to parse(),
    //! supportsModel(), parseFromFile(), or parseWithWeightDescriptors().
    //!
    virtual char const* const* getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept = 0;

    //!
    //! \brief Set the parser flags.
    //!
    //! The flags are listed in the OnnxParserFlag enum.
    //!
    //! \param OnnxParserFlag The flags used when parsing an ONNX model.
    //!
    //! \note This function will override the previous set flags, rather than bitwise ORing the new
    //! flag.
    //!
    //! \see getFlags()
    //!
    virtual void setFlags(OnnxParserFlags onnxParserFlags) noexcept = 0;

    //!
    //! \brief Get the parser flags. Defaults to 0.
    //!
    //! \return The parser flags as a bitmask.
    //!
    //! \see setFlags()
    //!
    virtual OnnxParserFlags getFlags() const noexcept = 0;

    //!
    //! \brief clear a parser flag.
    //!
    //! clears the parser flag from the enabled flags.
    //!
    //! \see setFlags()
    //!
    virtual void clearFlag(OnnxParserFlag onnxParserFlag) noexcept = 0;

    //!
    //! \brief Set a single parser flag.
    //!
    //! Add the input parser flag to the already enabled flags.
    //!
    //! \see setFlags()
    //!
    virtual void setFlag(OnnxParserFlag onnxParserFlag) noexcept = 0;

    //!
    //! \brief Returns true if the parser flag is set
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    virtual bool getFlag(OnnxParserFlag onnxParserFlag) const noexcept = 0;

    //!
    //!\brief Return the i-th output ITensor object for the ONNX layer "name".
    //!
    //! Return the i-th output ITensor object for the ONNX layer "name".
    //! If "name" is not found or i is out of range, return nullptr.
    //! In the case of multiple nodes sharing the same name this function will return
    //! the output tensors of the first instance of the node in the ONNX graph.
    //!
    //! \param name The name of the ONNX layer.
    //!
    //! \param i The index of the output. i must be in range [0, layer.num_outputs).
    //!
    virtual mgxinfer1::ITensor const* getLayerOutputTensor(char const* name, int64_t i) = 0;
};

class Parser : public IParser
{
    public:
    bool parse(void const* serialized_onnx_model,
               size_t serialized_onnx_model_size,
               const char* model_path = nullptr) override
    {
        // TODO complete implementation, figure out what model_path does
        migraphx::onnx_options opts;
        network_.setProgram(std::make_shared<migraphx::program>(
            migraphx::parse_onnx_buffer(serialized_onnx_model, serialized_onnx_model_size, opts)));
        return true;
        pass("Not Implemented", true);
    }

    bool parseFromFile(const char* onnxModelFile, int verbosity) override
    {
        // NOTE In TRT the parser uses the network's layer creation methods to construct the
        // network. Due to how different the approach is MGX, (for now atleast) the parser will hand
        // the network the program, which required expanding the INetwork signature.
        // TODO error handling
        network_.setProgram(
            std::make_shared<migraphx::program>(migraphx::parse_onnx(onnxModelFile)));
        return true;
    }

    bool supportsModel(void const* serialized_onnx_model,
                       size_t serialized_onnx_model_size,
                       SubGraphCollection_t& sub_graph_collection,
                       const char* model_path = nullptr) override
    {
        pass("Not Implemented", true);
    }

    bool parseWithWeightDescriptors(void const* serialized_onnx_model,
                                    size_t serialized_onnx_model_size) override
    {
        pass("Not Implemented", true);
    }

    bool supportsOperator(const char* op_name) const override { pass("Not Implemented", true); }

    int getNbErrors() const override
    { // TODO implement actual error counting
        return 0;
    }

    IParserError const* getError(int index) const override { pass("Not Implemented", true); }

    void clearErrors() override { pass("Not Implemented", true); }

    char const* const* getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept override
    {
        pass("Not Implemented", true);
    }

    void setFlags(OnnxParserFlags onnxParserFlags) noexcept override
    {
        pass("Not Implemented", true);
    }

    OnnxParserFlags getFlags() const noexcept override { pass("Not Implemented", true); }

    void clearFlag(OnnxParserFlag onnxParserFlag) noexcept override
    {
        pass("Not Implemented", true);
    }

    void setFlag(OnnxParserFlag onnxParserFlag) noexcept override { pass("Not Implemented", true); }

    bool getFlag(OnnxParserFlag onnxParserFlag) const noexcept override
    {
        pass("Not Implemented", true);
    }

    mgxinfer1::ITensor const* getLayerOutputTensor(char const* name, int64_t i) override
    {
        pass("Not Implemented", true);
    }

    private:
    mgxinfer1::INetworkDefinition& network_;
    mgxinfer1::ILogger& logger_;

    Parser(mgxinfer1::INetworkDefinition& network, mgxinfer1::ILogger& logger)
        : network_{network}, logger_{logger}
    {
    }

    friend IParser* createParser(mgxinfer1::INetworkDefinition&, mgxinfer1::ILogger&);
};

//!
//! \brief Create a new parser object
//!
//! \param network The network definition that the parser will write to
//! \param logger The logger to use
//! \return a new parser object or NULL if an error occurred
//!
//! Any input dimensions that are constant should not be changed after parsing,
//! because correctness of the translation may rely on those constants.
//! Changing a dynamic input dimension, i.e. one that translates to -1 in
//! TensorRT, to a constant is okay if the constant is consistent with the model.
//! Each instance of the parser is designed to only parse one ONNX model once.
//!
//! \see IParser
//!
inline IParser* createParser(mgxinfer1::INetworkDefinition& network, mgxinfer1::ILogger& logger)
{
    return new Parser{network, logger};
}

} // namespace mgxonnxparser
