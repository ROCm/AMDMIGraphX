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
    auto comp_shape    = arg.get_shape();
    auto out_lens      = comp_shape.lens();
    auto dim_0         = out_lens.size() - 2;
    auto dim_1         = out_lens.size() - 1;
    std::size_t lda    = comp_shape.strides()[dim_0];
    std::size_t m_size = out_lens[dim_0] * out_lens[dim_1];
    visit_all(result, arg)([&](auto output, auto input) {
        std::size_t nelements = comp_shape.elements();
        auto* out_ptr         = device_cast(output.data());
        auto* in_ptr          = device_cast(input.data());
        visit_tensor_size(out_lens.size(), [&](auto out_dim) {
            hip_tensor_descriptor<out_dim> desc(comp_shape);
            gs_launch(stream, nelements, 256)([=](auto ii) {
                const size_t nb    = 4;
                auto idx           = desc.multi(ii);
                std::size_t i_m    = idx[dim_1];
                std::size_t i_k    = idx[dim_0];
                std::size_t offset = ii / m_size * m_size;
                out_ptr[i_k % nb + (i_m + (i_k / nb) * lda) * nb + offset] =
                    in_ptr[i_m + i_k * lda + offset];
            });
        });
    });
}

void pack_b(hipStream_t stream, const argument& result, const argument& arg)
{
    auto trans_shape = arg.get_shape();
    auto out_lens    = trans_shape.lens();
    auto dim_0       = trans_shape.lens().size() - 2;
    auto dim_1       = trans_shape.lens().size() - 1;
    std::size_t ldb  = trans_shape.strides()[dim_1];

    auto wrap_lens = out_lens;
    std::swap(wrap_lens[dim_0], wrap_lens[dim_1]);
    shape comp_shape{trans_shape.type(), wrap_lens};
    std::size_t m_size = out_lens[dim_0] * out_lens[dim_1];
    visit_all(result, arg)([&](auto output, auto input) {
        std::size_t nelements = comp_shape.elements();
        auto* out_ptr         = device_cast(output.data());
        auto* in_ptr          = device_cast(input.data());
        visit_tensor_size(out_lens.size(), [&](auto out_dim) {
            hip_tensor_descriptor<out_dim> desc(comp_shape);
            gs_launch(stream, nelements, 256)([=](auto ii) {
                const size_t nb    = 4;
                auto idx           = desc.multi(ii);
                std::size_t i_n    = idx[dim_1];
                std::size_t i_k    = idx[dim_0];
                std::size_t offset = ii / m_size * m_size;
                out_ptr[i_k % nb + (i_n + (i_k / nb) * ldb) * nb + offset] =
                    in_ptr[i_n + i_k * ldb + offset];
            });
        });
    });
}

void sync_stream(hipStream_t stream) { hipStreamSynchronize(stream); }

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
