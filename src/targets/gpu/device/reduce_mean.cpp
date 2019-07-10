#include <migraphx/gpu/device/reduce_mean.hpp>
#include <migraphx/gpu/device/reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void reduce_mean(hipStream_t stream, const argument& result, const argument& arg)
{
    std::size_t item_num = arg.get_shape().elements() / result.get_shape().elements();
    reduce(stream, result, arg, sum{}, 0, id{}, mean{item_num});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
