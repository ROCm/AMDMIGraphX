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
    std::size_t batch_size = std::accumulate(
        lens.begin(), lens.begin() + axis, std::size_t{1}, std::multiplies<std::size_t>());
    std::size_t n_dims = std::accumulate(
        lens.begin() + axis, lens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    migraphx::shape comp_shape{output_shape.type(), {batch_size, n_dims}};

    visit_all(args.back(), args.front())([&](auto output, auto input) {
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr      = device_cast(output.data());

        // each thread is for one item in the batch
        gs_launch(stream, batch_size)([=](auto i) {
            std::size_t row_start = i * n_dims;
            // get max
            auto batch_max = input_ptr[row_start];
            for(std::size_t j = 1; j < n_dims; ++j)
            {
                auto ind  = row_start + j;
                batch_max = std::max(to_hip_type(batch_max), to_hip_type(input_ptr[ind]));
            }

            for(std::size_t j = 0; j < n_dims; ++j)
            {
                auto ind        = row_start + j;
                output_ptr[ind] = input_ptr[ind] - batch_max;
            }

            auto batch_sum = ::exp(to_hip_type(output_ptr[row_start]));
            for(std::size_t j = 1; j < n_dims; ++j)
            {
                auto ind = row_start + j;
                batch_sum += ::exp(to_hip_type(output_ptr[ind]));
            }
            batch_sum = ::log(to_hip_type(batch_sum));

            for(std::size_t j = 0; j < n_dims; ++j)
            {
                auto ind = row_start + j;
                output_ptr[ind] -= batch_sum;
            }
        });
    });

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
