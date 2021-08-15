#include "migraphx/gpu/device/shape.hpp"
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/topk.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/visit.hpp>
#include <migraphx/gpu/device/heap.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument topk(hipStream_t stream,
              argument val_res,
              argument ind_res,
              argument arg,
              int64_t k,
              int64_t axis,
              bool largest)
{
    auto in_s       = arg.get_shape();
    auto in_lens    = in_s.lens();
    auto out_s      = val_res.get_shape();
    auto axis_dim   = in_s.lens()[axis];
    auto comp_lens  = in_lens;
    comp_lens[axis] = 1;
    shape comp_s{in_s.type(), comp_lens};
    std::size_t elem_num = comp_s.elements();

    hip_visit_all(val_res, arg, out_s, in_s, comp_s)(
        [&](auto out_val, auto input, auto oss, auto iss, auto css) {
            auto* data      = device_cast(input.data());
            auto* out       = device_cast(out_val.data());
            auto* const ind = ind_res.cast<int64_t>();
            gs_launch(stream, elem_num)([=](auto i) __device__ {
                auto idx = css.multi(i);

                auto in_idx = [&](int ii) {
                    auto iidx  = idx;
                    iidx[axis] = ii;
                    return iss.index(iidx);
                };

                auto out_idx = [&](int ii) {
                    auto iidx  = idx;
                    iidx[axis] = ii;
                    return oss.index(iidx);
                };

                auto compare = [=](auto ii, auto jj) {
                    return largest ? std::less<>{}(data[in_idx(ii)], data[in_idx(jj)])
                                   : std::greater<>{}(data[in_idx(ii)], data[in_idx(jj)]);
                };

                for(int j = 0; j < k; ++j)
                {
                    ind[out_idx(j)] = j;
                }

                auto hp = make_heap(ind, k, out_idx, compare);
                for(int j = k; j < axis_dim; ++j)
                {
                    hp.update(j);
                }
                hp.sort();

                for(int j = 0; j < k; ++j)
                {
                    out[out_idx(j)] = data[in_idx(ind[out_idx(j)])];
                }
            });
        });

    return argument({val_res, ind_res});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
