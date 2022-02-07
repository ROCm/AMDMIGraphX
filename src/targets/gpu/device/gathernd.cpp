#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/gpu/device/gathernd.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument gathernd(
    hipStream_t stream, argument result, argument arg0, argument arg1, const int& batch_dims)
{
    std::cout << "Before decs" << std::endl;
    auto indices_shape_lens = arg1.get_shape().lens();
    auto data_s_lens = arg0.get_shape().lens();
    auto num_slice_dims = indices_shape_lens.back();
    std::size_t num_slices = std::accumulate(indices_shape_lens.begin(), indices_shape_lens.end() - 1, 1, std::multiplies<std::size_t>());
    std::size_t slice_size = std::accumulate(data_s_lens.begin() + num_slice_dims + batch_dims, data_s_lens.end(), 1, std::multiplies<std::size_t>());
    const std::size_t num_batches = std::accumulate(data_s_lens.begin(), data_s_lens.begin() + batch_dims, 1, std::multiplies<std::size_t>());
    const std::size_t data_batch_stride = std::accumulate(data_s_lens.begin() + batch_dims, data_s_lens.end(), 1, std::multiplies<std::size_t>());
    const auto num_slices_per_batch = num_slices / num_batches;
    std::cout << "After" << std::endl;
    //hip_visit_all(arg0, arg0.get_shape())([&](auto data, auto data_shape) {
    //   hip_visit_all(arg1, arg1.get_shape())([&](auto indices, auto indices_shape) {
    //        result.visit([&](auto output_view) {
    //            hip_visit_views(output_view)([&](auto output) {
    /*
    visit_all(result, arg0)([&](auto output, auto data) {
        arg1.visit([&](auto indices) {
            hip_visit_views(indices.get_shape())([&](auto indices_shape) {
                hip_visit_views(data.get_shape())([&](auto data_shape) {
    */
    /*
    hip_visit_all(arg0, arg0.get_shape())([&](auto data, auto data_shape) {
       hip_visit_all(arg1, arg1.get_shape())([&](auto indices, auto indices_shape) {
           hip_visit_all(result, result.get_shape())([&](auto output, auto) {
    */
   
    visit_all(result, arg0)([&](auto output, auto data) {
        arg1.visit([&](auto indices) {
            hip_visit_views(data.get_shape())([&](auto data_shape) {
                //hip_visit_views(data_)([&](auto data) {
            //hip_visit_views(indices_s, data_s)([&](auto indices_shape, auto data_shape) {
                //hip_visit_views(data_s)([&](auto data_shape) {
                    auto data_shape_lens = data_shape.lens;
                    
                    std::size_t* sizes_from_slice_dims_host = new std::size_t[num_slice_dims];
                    auto running_product = slice_size;
                    for (std::size_t i = 0; i < num_slice_dims; ++i) {
                        sizes_from_slice_dims_host[num_slice_dims - 1 - i] = running_product;
                        running_product *= data_shape_lens[batch_dims + num_slice_dims - 1 - i];
                    }
                    std::cout << "after 1" << std::endl;
                    std::size_t* input_slice_offsets_host = new std::size_t[num_slices];
                    auto* input_slice_offsets = device_cast(input_slice_offsets_host);
                    auto* sizes_from_slice_dims = device_cast(sizes_from_slice_dims_host);
                    //std::shared_ptr<std::size_t> input_slices_offsets

                    // Compute Slice Offsets
                    const auto* indices_ptr = device_cast(indices.data());
                    gs_launch(stream, num_slices)([&](auto i) __device__ {
                        const std::size_t batch_idx = i / num_slices_per_batch;
                        const std::size_t base_offset = batch_idx * data_batch_stride;

                        auto* slice_indices = indices_ptr + (i * num_slice_dims);
                        std::size_t relative_slice_offset = 0;
                        for (size_t idx = 0; idx < num_slice_dims; ++idx) {
                            auto index = slice_indices[idx];
                            const std::size_t input_dim_idx = batch_dims + idx;
                            if (index < 0) 
                                index += data_shape_lens[input_dim_idx];
                            relative_slice_offset += index * sizes_from_slice_dims[idx];
                        }

                        input_slice_offsets[i] = base_offset + relative_slice_offset;
                    });

                    for (std::size_t i = 0; i < num_slices; ++i)
                        std::cout << input_slice_offsets_host[i] << ", ";
                    std::cout << std::endl;
                    
                    std::cout << "after 2" << std::endl;
                    auto* output_ptr = device_cast(output.data());
                    const auto* data_ptr = device_cast(data.data());
                    gs_launch(stream, num_slices * slice_size)([&](auto i) __device__  {
                        auto slice_offset = input_slice_offsets[i / slice_size];
                        output_ptr[i] = data_ptr[slice_offset + i % slice_size];
                    });
                    std::cout << "after 3" << std::endl;
                    //delete[] sizes_from_slice_dims;
                    //delete[] input_slice_offsets;
                //});
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
