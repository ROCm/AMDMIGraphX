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

void pack_a(hipStream_t stream, const argument& result, const argument& arg)
{
    auto output_shape = result.get_shape();
    auto out_lens = output_shape.lens();
    auto dim_0        = out_lens.size() - 2;
    auto dim_1        = out_lens.size() - 1;
    std::size_t lda   = output_shape.strides()[dim_0];
    std::size_t m_size = out_lens[dim_0] * out_lens[dim_1];
    visit_all(result, arg)([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        auto* out_ptr         = device_cast(output.data());
        auto* in_ptr          = device_cast(input.data());
        visit_tensor_size(out_lens.size(), [&](auto out_dim) {
            hip_tensor_descriptor<out_dim> desc(output_shape);
            gs_launch(stream, nelements)([=](auto ii) {
                const size_t nb                                   = 4;
                auto idx                                          = desc.multi(ii);
                std::size_t i_m                                   = idx[dim_1];
                std::size_t i_k                                   = idx[dim_0];
                std::size_t offset = ii / m_size * m_size;
                out_ptr[i_k % nb + (i_m + (i_k / nb) * lda) * nb + offset] = in_ptr[i_m + i_k * lda + offset];
            });
        });
    });
}

void pack_b(hipStream_t stream, const argument& result, const argument& arg)
{
    auto output_shape = result.get_shape();
    auto out_lens = output_shape.lens();
    auto dim_0        = output_shape.lens().size() - 2;
    auto dim_1        = output_shape.lens().size() - 1;
    std::size_t ldb   = output_shape.strides()[dim_1];
    std::size_t m_size = out_lens[dim_0] * out_lens[dim_1];
    visit_all(result, arg)([&](auto output, auto input) {
        std::size_t nelements = output_shape.elements();
        auto* out_ptr         = device_cast(output.data());
        auto* in_ptr          = device_cast(input.data());
        visit_tensor_size(out_lens.size(), [&](auto out_dim) {
            hip_tensor_descriptor<out_dim> desc(output_shape);
            gs_launch(stream, nelements)([=](auto ii) {
                const size_t nb                                   = 4;
                auto idx                                          = desc.multi(ii);
                std::size_t i_n                                   = idx[1];
                std::size_t i_k                                   = idx[0];
                std::size_t offset = ii / m_size * m_size;
                out_ptr[i_k % nb + (i_n + (i_k / nb) * ldb) * nb + offset] = in_ptr[i_n + i_k * ldb + offset];
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
