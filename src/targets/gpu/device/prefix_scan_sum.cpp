#include <migraphx/gpu/device/prefix_scan_sum.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template<class F>
MIGRAPHX_DEVICE_CONSTEXPR void visit_bool(bool b, F f)
{
    if (b)
        f(std::true_type{});
    else
        f(std::false_type{});
}

void prefix_scan_sum(hipStream_t stream,
                     const argument& result,
                     const argument& arg,
                     int32_t axis,
                     bool exclusive,
                     bool reverse)
{
    const index_int block_size = 256;
    const index_int n          = arg.get_shape().lens()[axis];
    auto rlens                 = result.get_shape().lens();
    rlens[axis]                = 1;

    hip_visit_all(result, arg, result.get_shape().with_lens(rlens))(
        [=](auto output, auto input, auto rshape) {
            gs_launch(stream, rshape.elements() * block_size, block_size)(
                [=](auto i, auto idx) __device__ {
                    const auto ridx  = rshape.multi(i / block_size);
                    auto compute_idx = [&](auto j) {
                        auto k  = ridx;
                        k[axis] = j;
                        return k;
                    };
                    visit_bool(reverse, [&](auto reverse_c) {
                        if constexpr(reverse_c)
                        {
                            block_scan<block_size>(
                                idx,
                                sum{},
                                0,
                                n,
                                reverse_scan(n,
                                            [&](auto j) {
                                                return input[compute_idx(j)];
                                            }),
                                reverse_scan(n, [&](auto j, auto x) { 
                                    visit_bool(exclusive, [&](auto exclusive_c) {
                                        if constexpr(exclusive_c)
                                        {
                                            if (j == n - 1)
                                                output[compute_idx(j)] = 0;
                                            if (j > 0)
                                                output[compute_idx(j - 1)] = x;
                                        }
                                        else 
                                        {
                                            output[compute_idx(j)] = x;
                                        }
                                    });
                                }));
                        }
                        else 
                        {
                            block_scan<block_size>(
                                idx,
                                sum{},
                                0,
                                n,
                                [&](auto j) {
                                    return input[compute_idx(j)];
                                },
                                [&](auto j, auto x) { 
                                    visit_bool(exclusive, [&](auto exclusive_c) {
                                        if constexpr(exclusive_c)
                                        {
                                            auto k = j + 1;
                                            if (j == 0)
                                                output[compute_idx(0)] = 0;
                                            if (k < n)
                                                output[compute_idx(k)] = x;
                                        }
                                        else 
                                        {
                                            output[compute_idx(j)] = x;
                                        }
                                    });
                                });
                        }
                    });
                    
                });
        });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
