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

template <class Iterator, class T>
constexpr Iterator upper_bound(Iterator first, Iterator last, const T& value)
{
    Iterator it;
    typename std::iterator_traits<Iterator>::difference_type count;
    typename std::iterator_traits<Iterator>::difference_type step;
    count = std::distance(first, last);

    while(count > 0)
    {
        it   = first;
        step = count / 2;
        std::advance(it, step);
        if(!(value < *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
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
                    auto idx       = output.get_shape().multi(i);
                    auto cdf_begin = cdf.begin() + (idx.front() * class_size);
                    auto cdf_end   = cdf_begin + class_size;
                    auto sample_iter =
                        upper_bound(cdf_begin, cdf_end, dist[i] * *(std::prev(cdf_end)));
                    output[i] = std::distance(cdf_begin, sample_iter);
                });
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
