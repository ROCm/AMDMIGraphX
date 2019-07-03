#include <migraphx/gpu/device/reduce_mean.hpp>
#include <migraphx/gpu/device/reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void reduce_mean(hipStream_t stream, const argument& result, const argument& arg)
{
    std::size_t batch_item_num = arg.get_shape().elements() / result.get_shape().elements();
    float factor = 1.0f / batch_item_num;
    reduce(stream, result, arg, sum{}, 0, id{}, scale{factor});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
