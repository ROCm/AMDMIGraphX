#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/gather.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument gather(hipStream_t stream,
                const migraphx::shape& output_shape,
                std::vector<migraphx::argument> args,
                int axis)
{
    int axis_index = (axis < 0) ? (axis + output_shape.lens().size()) : axis;
    visit_all(args.back(), args[0])([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        args[1].visit([&](auto indices) {
            const auto* indices_ptr = device_cast(indices.data());
            auto* outptr            = device_cast(output.data());
            const auto* inptr       = device_cast(input.data());
            if(output_shape.scalar())
            {
                gs_launch(stream,
                          1)([=](auto i) { outptr[i] = inptr[static_cast<int>(indices_ptr[0])]; });
            }
            else
            {
                visit_tensor_size(output_shape.lens().size(), [&](auto n_out_dim) {
                    visit_tensor_size(args[0].get_shape().lens().size(), [&](auto n_in_dim) {
                        hip_tensor_descriptor<n_in_dim> desc_input(input.get_shape());
                        hip_tensor_descriptor<n_out_dim> desc_output(output.get_shape());
                        if(args[1].get_shape().scalar())
                        {
                            gs_launch(stream, nelements)([=](auto ii) {
                                auto out_idx = desc_output.multi(ii);
                                auto in_idx  = desc_input.multi(0);
                                for(int i = 0; i < axis_index; ++i)
                                {
                                    in_idx[i] = out_idx[i];
                                }
                                in_idx[axis_index] = indices_ptr[0];
                                for(int i = axis_index + 1; i < n_in_dim; ++i)
                                {
                                    in_idx[i] = out_idx[i - 1];
                                }
                                outptr[ii] = inptr[desc_input.linear(in_idx)];
                            });
                        }
                        else
                        {
                            visit_tensor_size(
                                args[1].get_shape().lens().size(), [&](auto n_ind_dim) {
                                    hip_tensor_descriptor<n_ind_dim> desc_ind(args[1].get_shape());
                                    gs_launch(stream, nelements)([=](auto ii) {
                                        auto out_idx = desc_output.multi(ii);
                                        auto in_idx  = desc_input.multi(0);
                                        for(int i = 0; i < axis_index; ++i)
                                        {
                                            in_idx[i] = out_idx[i];
                                        }
                                        auto ind_idx = desc_ind.multi(0);
                                        for(int i = 0; i < n_ind_dim; ++i)
                                        {
                                            ind_idx[i] = out_idx[i + axis_index];
                                        }
                                        in_idx[axis_index] = indices_ptr[desc_ind.linear(ind_idx)];
                                        for(int i = axis_index + 1; i < n_in_dim; ++i)
                                        {
                                            in_idx[i] = out_idx[i + n_ind_dim - 1];
                                        }
                                        outptr[ii] = inptr[desc_input.linear(in_idx)];
                                    });
                                });
                        }
                    });
                });
            }
        });
    });

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
