#include <migraphx/gpu/device/reduce_prod.hpp>
#include <migraphx/gpu/device/reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void reduce_prod(hipStream_t stream, const argument& result, const argument& arg)
{

    reduce(stream, result, arg, product{}, 1, id{}, id{});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
