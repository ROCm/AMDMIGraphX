#pragma once

#include <cstddef>
#include <cstdint>

namespace mgxinfer1 {

//! char_t is the type used by TensorRT to represent all valid characters.
using char_t = char;

//! AsciiChar is the type used by TensorRT to represent valid ASCII characters.
//! This type is widely used in automotive safety context.
using AsciiChar = char_t;

//!
//! \enum DataType
//! \brief The type of weights and tensors.
//!
enum class DataType : int32_t
{
    //! 32-bit floating point format.
    kFLOAT = 0,

    //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
    kHALF = 1,

    //! Signed 8-bit integer representing a quantized floating-point value.
    kINT8 = 2,

    //! Signed 32-bit integer format.
    kINT32 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4,

    //! Unsigned 8-bit integer format.
    //! Cannot be used to represent quantized floating-point values.
    //! Use the IdentityLayer to convert kUINT8 network-level inputs to {kFLOAT, kHALF} prior
    //! to use with other TensorRT layers, or to convert intermediate output
    //! before kUINT8 network-level outputs from {kFLOAT, kHALF} to kUINT8.
    //! kUINT8 conversions are only supported for {kFLOAT, kHALF}.
    //! kUINT8 to {kFLOAT, kHALF} conversion will convert the integer values
    //! to equivalent floating point values.
    //! {kFLOAT, kHALF} to kUINT8 conversion will convert the floating point values
    //! to integer values by truncating towards zero. This conversion has undefined behavior for
    //! floating point values outside the range [0.0F, 256.0F) after truncation.
    //! kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
    kUINT8 = 5,

    //! Signed 8-bit floating point with
    //! 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
    kFP8 = 6,

    //! Brain float -- has an 8 bit exponent and 8 bit significand.
    kBF16 = 7,

    //! Signed 64-bit integer type.
    kINT64 = 8,

    //! Signed 4-bit integer type.
    kINT4 = 9,
};

//!
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! TensorRT can also return an "invalid dims" structure. This structure is
//! represented by nbDims == -1 and d[i] == 0 for all i.
//!
//! TensorRT can also return an "unknown rank" dims structure. This structure is
//! represented by nbDims == -1 and d[i] == -1 for all i.
//!
class Dims64
{
    public:
    //! The maximum rank (number of dimensions) supported for a tensor.
    static constexpr int32_t MAX_DIMS{8};

    //! The rank (number of dimensions).
    int32_t nbDims;

    //! The extent of each dimension.
    int64_t d[MAX_DIMS];
};

//!
//! Alias for Dims64.
//!
using Dims = Dims64;

//!
//! \enum TensorFormat
//!
//! \brief Format of the input/output tensors.
//!
//! This enum is used by both plugins and network I/O tensors.
//!
//! \see IPluginV2::supportsFormat(), safe::ICudaEngine::getBindingFormat()
//!
//! Many of the formats are **vector-major** or **vector-minor**. These formats specify
//! a <em>vector dimension</em> and <em>scalars per vector</em>.
//! For example, suppose that the tensor has has dimensions [M,N,C,H,W],
//! the vector dimension is C and there are V scalars per vector.
//!
//! * A **vector-major** format splits the vectorized dimension into two axes in the
//!   memory layout. The vectorized dimension is replaced by an axis of length ceil(C/V)
//!   and a new dimension of length V is appended. For the example tensor, the memory layout
//!   is equivalent to an array with dimensions [M][N][ceil(C/V)][H][W][V].
//!   Tensor coordinate (m,n,c,h,w) maps to array location [m][n][c/V][h][w][c\%V].
//!
//! * A **vector-minor** format moves the vectorized dimension to become the last axis
//!   in the memory layout. For the example tensor, the memory layout is equivalent to an
//!   array with dimensions [M][N][H][W][ceil(C/V)*V]. Tensor coordinate (m,n,c,h,w) maps
//!   array location subscript [m][n][h][w][c].
//!
//! In interfaces that refer to "components per element", that's the value of V above.
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide.
//! https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc
//!
enum class TensorFormat : int32_t
{
    //! Memory layout is similar to an array in C or C++.
    //! The stride of each dimension is the product of the dimensions after it.
    //! The last dimension has unit stride.
    //!
    //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
    kLINEAR = 0,

    //! Vector-major format with two scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires FP16 or BF16 and at least three dimensions.
    kCHW2 = 1,

    //! Vector-minor format with eight scalars per vector.
    //! Vector dimension is third to last.
    //! This format requires FP16 or BF16 and at least three dimensions.
    kHWC8 = 2,

    //! Vector-major format with four scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires INT8 or FP16 and at least three dimensions.
    //! For INT8, the length of the vector dimension must be a build-time constant.
    //!
    //! Deprecated usage:
    //!
    //! If running on the DLA, this format can be used for acceleration
    //! with the caveat that C must be less than or equal to 4.
    //! If used as DLA input and the build option kGPU_FALLBACK is not specified,
    //! it needs to meet line stride requirement of DLA format. Column stride in
    //! bytes must be a multiple of 64 on Orin.
    kCHW4 = 3,

    //! Vector-major format with 16 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires INT8 or FP16 and at least three dimensions.
    //!
    //! For DLA usage, this format maps to the native feature format for FP16,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW16 = 4,

    //! Vector-major format with 32 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires at least three dimensions.
    //!
    //! For DLA usage, this format maps to the native feature format for INT8,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW32 = 5,

    //! Vector-minor format with eight scalars per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP16 or BF16 and at least four dimensions.
    kDHWC8 = 6,

    //! Vector-major format with 32 scalars per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP16 or INT8 and at least four dimensions.
    kCDHW32 = 7,

    //! Vector-minor format where channel dimension is third to last and unpadded.
    //!
    //! This format requires either FP32 or UINT8 and at least three dimensions.
    kHWC = 8,

    //! DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
    //! always has unit stride. The stride for stepping along the H axis is
    //! rounded up to 64 bytes.
    //!
    //! The memory layout is equivalent to a C array with dimensions
    //! [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
    //! 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c][h][w].
    kDLA_LINEAR = 9,

    //! DLA image format. For a tensor with dimension {N, C, H, W} the C axis
    //! always has unit stride. The stride for stepping along the H axis is rounded up
    //! to 64 bytes on Orin. C can only be 1, 3 or 4.
    //! If C == 1, it will map to grayscale format.
    //! If C == 3 or C == 4, it will map to color image format. And if C == 3,
    //! the stride for stepping along the W axis needs to be padded to 4 in elements.
    //!
    //! When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
    //! where elementSize is 2 for FP16
    //! and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
    //! subscript [n][h][w][c].
    kDLA_HWC4 = 10,

    //! Vector-minor format with 16 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This requires FP16 and at least three dimensions.
    kHWC16 = 11,

    //! Vector-minor format with one scalar per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP32 and at least four dimensions.
    kDHWC = 12
};

//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, refitter and runtime.
//!
//! The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects
//! created through that interface. The logger must be valid until all objects created are released.
//!
//! The Logger object implementation must be thread safe. All locking and synchronization is pushed
//! to the interface implementation and TensorRT does not hold any synchronization primitives when
//! calling the interface functions.
//!
class ILogger
{
    public:
    //!
    //! \enum Severity
    //!
    //! \brief The severity corresponding to a log message.
    //!
    enum class Severity : int32_t
    {
        //! An internal error has occurred. Execution is unrecoverable.
        kINTERNAL_ERROR = 0,
        //! An application error has occurred.
        kERROR = 1,
        //! An application error has been discovered, but TensorRT has recovered or fallen back to a
        //! default.
        kWARNING = 2,
        //!  Informational messages with instructional information.
        kINFO = 3,
        //!  Verbose messages with debugging information.
        kVERBOSE = 4,
    };

    //!
    //! \brief A callback implemented by the application to handle logging messages;
    //!
    //! \param severity The severity of the message.
    //! \param msg A null-terminated log message.
    //!
    //! \warning Loggers used in the safety certified runtime must set a maximum message length and
    //! truncate
    //!          messages exceeding this length. It is up to the implementer of the derived class to
    //!          define a suitable limit that will prevent buffer overruns, resource exhaustion, and
    //!          other security vulnerabilities in their implementation. The TensorRT safety
    //!          certified runtime will never emit messages longer than 1024 bytes.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from
    //!   multiple threads
    //!                  when multiple execution contexts are used during runtime, or if the same
    //!                  logger is used for multiple runtimes, builders, or refitters.
    //!
    virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;

    ILogger()          = default;
    virtual ~ILogger() = default;

    protected:
    // @cond SuppressDoxyWarnings
    ILogger(ILogger const&)              = default;
    ILogger(ILogger&&)                   = default;
    ILogger& operator=(ILogger const&) & = default;
    ILogger& operator=(ILogger&&) &      = default;
    // @endcond
};

//!
//! \enum TensorIOMode
//!
//! \brief Definition of tensor IO Mode.
//!
enum class TensorIOMode : int32_t
{
    //! Tensor is not an input or output.
    kNONE = 0,

    //! Tensor is input to the engine.
    kINPUT = 1,

    //! Tensor is output by the engine.
    kOUTPUT = 2
};

} // namespace mgxinfer1
