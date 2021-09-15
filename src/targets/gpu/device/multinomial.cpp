#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/multinomial.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T1, class T2>
void __device__
set_output(T1& cdf, T1& dist, T2& output, const size_t& i, size_t& j, const size_t& n)
{
    for(; j < n; ++j)
    {
        if(cdf[j] > (dist[i] * cdf[n - 1]))
        {
            output[i] = j;
            return;
        }
    }
}

void multinomial(hipStream_t stream,
                 const argument& result,
                 const argument& arg0,
                 const argument& arg1)
{
    size_t batch_size  = arg0.get_shape().lens().front();
    size_t class_size  = arg0.get_shape().lens().back();
    size_t sample_size = result.get_shape().lens().back();

    hip_visit_all(arg0, arg1)([&](auto cdf, auto dist) {
        result.visit([&](auto out) {
            hip_visit_views(out)([&](auto output) {
                gs_launch(stream, batch_size * sample_size)([=](auto i) __device__ {
                    auto idx = output.get_shape().multi(i);
                    size_t j = idx.front() * class_size;
                    size_t n = (idx.front() + 1) * class_size;
                    set_output(cdf, dist, output, i, j, n);
                });
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
