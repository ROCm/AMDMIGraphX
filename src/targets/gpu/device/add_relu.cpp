#include <migraphx/gpu/device/add_relu.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void mul_add_relu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3)
{
    nary(stream, result, arg1, arg2, arg3)(
        [](auto x, auto a, auto b) { return std::max<decltype(a*x + b)>(0, a*x + b); });
}

void add_relu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2)
{
    nary(stream, result, arg1, arg2)(
        [](auto x, auto y) { return std::max<decltype(x + y)>(0, x + y); });
}

void add_relu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3)
{
    nary(stream, result, arg1, arg2, arg3)(
        [](auto x, auto y, auto z) { return std::max<decltype(x + y + z)>(0, x + y + z); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
