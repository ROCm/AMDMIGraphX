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

argument logsoftmax(hipStream_t stream,
                    const migraphx::shape& output_shape,
                    std::vector<migraphx::argument> args,
                    int axis)
{

    auto lens              = output_shape.lens();
    auto num_in_batch = lens[axis];
    auto batch_lens = lens;
    batch_lens[axis] = 1;
    migraphx::shape batch_shape{output_shape.type(), batch_lens};

    visit_all(args.back(), args.front())([&](auto output, auto input) {
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr      = device_cast(output.data());
        visit_tensor_size(batch_shape.lens().size(), [&](auto n_dim) {
            hip_tensor_descriptor<n_dim> desc_batch(batch_shape);
            hip_tensor_descriptor<n_dim> desc_data(output_shape);

            // each thread is for one item in the batch
            gs_launch(stream, batch_shape.elements())([=](auto i) {
                auto batch_idx = desc_batch.multi(i);
                auto data_idx = batch_idx;

                // get max
                auto batch_max = input_ptr[desc_data.linear(batch_idx)];
                for(std::size_t j = 1; j < num_in_batch; ++j)
                {
                    data_idx[axis] = j;
                    size_t idx = desc_data.linear(data_idx);
                    batch_max = std::max(to_hip_type(batch_max), to_hip_type(input_ptr[idx]));
                }

                for(std::size_t j = 0; j < num_in_batch; ++j)
                {
                    data_idx[axis] = j;
                    size_t idx = desc_data.linear(data_idx);
                    output_ptr[idx] = input_ptr[idx] - batch_max;
                }

                auto batch_sum = ::exp(to_hip_type(output_ptr[desc_data.linear(batch_idx)]));
                for(std::size_t j = 1; j < num_in_batch; ++j)
                {
                    data_idx[axis] = j;
                    size_t idx = desc_data.linear(data_idx);
                    batch_sum += ::exp(to_hip_type(output_ptr[idx]));
                }
                batch_sum = ::log(to_hip_type(batch_sum));

                for(std::size_t j = 0; j < num_in_batch; ++j)
                {
                    data_idx[axis] = j;
                    size_t idx = desc_data.linear(data_idx);
                    output_ptr[idx] -= batch_sum;
                }
            });
        });
    });

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
