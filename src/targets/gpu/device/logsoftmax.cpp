#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/logsoftmax.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument logsoftmax(hipStream_t stream, argument result, argument arg, int axis)
{

    auto lens         = result.get_shape().lens();
    auto num_in_batch = lens[axis];
    auto batch_lens   = lens;
    batch_lens[axis]  = 1;
    shape batch_shape{result.get_shape().type(), batch_lens};

    hip_visit_all(result, arg, batch_shape)([&](auto output, auto input, auto batch) {

        // each thread is for one item in the batch
        gs_launch(stream, batch_shape.elements())([=](auto i) {
            auto batch_idx = batch.multi(i);
            auto data_idx  = batch_idx;

            // get max
            auto batch_max = input[batch_idx];
            for(std::size_t j = 1; j < num_in_batch; ++j)
            {
                data_idx[axis] = j;
                batch_max      = std::max(to_hip_type(batch_max), to_hip_type(input[data_idx]));
            }

            for(std::size_t j = 0; j < num_in_batch; ++j)
            {
                data_idx[axis]   = j;
                output[data_idx] = input[data_idx] - batch_max;
            }

            auto batch_sum = ::exp(to_hip_type(output[batch_idx]));
            for(std::size_t j = 1; j < num_in_batch; ++j)
            {
                data_idx[axis] = j;
                batch_sum += ::exp(to_hip_type(output[data_idx]));
            }
            batch_sum = ::log(to_hip_type(batch_sum));

            for(std::size_t j = 0; j < num_in_batch; ++j)
            {
                data_idx[axis] = j;
                output[data_idx] -= batch_sum;
            }
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
