#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/pack.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void pack_a(hipStream_t stream,
            const argument& result, const argument& arg)
{
    auto output_shape = result.get_shape();
    auto dim_0 = output_shape.lens().size() - 2;
    std::size_t ldb = output_shape.strides()[dim_0];
    visit_all(result, arg) ([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        auto* out_ptr           = device_cast(output.data());
        auto* in_ptr            = device_cast(input.data());
        visit_tensor_size(output_shape.lens().size(), [&](auto out_dim) {
            hip_tensor_descriptor<out_dim> desc(output_shape);
            gs_launch(stream, nelements)([=](auto ii) {
                const size_t nb = 4;
                auto idx        = desc.multi(ii);
                std::size_t i_m = idx[0];
                std::size_t i_k = idx[1];
                out_ptr[i_k % nb + (i_m + (i_k / nb) * ldb) * nb] = in_ptr[i_m + i_k * ldb];
            });
        });
    });
}

void pack_b(hipStream_t stream,
            const argument& result, const argument& arg) 
{
    auto output_shape = result.get_shape();
    auto dim_1 = output_shape.lens().size() - 1;
    std::size_t lda = output_shape.strides()[dim_1];
    visit_all(result, arg) ([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        auto* out_ptr           = device_cast(output.data());
        auto* in_ptr            = device_cast(input.data());
        visit_tensor_size(output_shape.lens().size(), [&](auto out_dim) {
            hip_tensor_descriptor<out_dim> desc(output_shape);
            gs_launch(stream, nelements)([=](auto ii) {
                const size_t nb = 4;
                auto idx        = desc.multi(ii);
                std::size_t i_n = idx[0];
                std::size_t i_k = idx[1];
                out_ptr[i_k % nb + (i_n + (i_k / nb) * lda) * nb] = in_ptr[i_n + i_k * lda];
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
