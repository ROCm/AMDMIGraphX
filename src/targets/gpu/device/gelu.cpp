#include <migraphx/gpu/device/gelu.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <cmath>

// GELU (Gaussian Error Linear Unit) activation function

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// A heuristic approximation of the GELU function:
// x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
template <class T>
auto gelu_fn(T x) __device__
{
    return x * 0.5 * (1 + ::erf(x * M_SQRT1_2));
}

// the magic number 0.044715 appears to originate with the BERT model paper by Jacob Devlin,
// Ming-Wei Chang, Kenton Lee, Kristina Toutanova
// The formula is a heuristic approximation of the GELU function:
// 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))
template <class T>
auto gelu_fn_new(T x) __device__
{
    return 0.5 * x * (1 + tanh(sqrt(M_2_PI) * (x + 0.044715 * x * x * x)));
}

void gelu(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ { return gelu_fn(to_hip_type(x)); });
}

void gelu_new(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ { return gelu_fn_new(to_hip_type(x)); });
}

void add_gelu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ {
        auto sum = to_hip_type(x + y);
        return gelu_fn(sum);
    });
}

void add_gelu_new(hipStream_t stream,
                  const argument& result,
                  const argument& arg1,
                  const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ {
        auto sum = to_hip_type(x + y);
        return gelu_fn(sum);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
