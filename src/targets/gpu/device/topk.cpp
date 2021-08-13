#include "migraphx/gpu/device/shape.hpp"
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/topk.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/visit.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
__device__ inline void swap(T& v1, T& v2)
{
    T v = v1;
    v1 = v2;
    v2 = v;
}

template <class IndIndex, class Compare>
__device__ inline void heap_heapify(int64_t* const ind,
                                    int n,
                                    int index,
                                    IndIndex ind_idx,
                                    Compare comp)
{
    while(index < n)
    {
        auto pre_index = index;
        int l          = 2 * index + 1;
        int r          = 2 * index + 2;

        if(l < n && comp(ind[ind_idx(l)], ind[ind_idx(index)]))
        {
            index = l;
        }

        if(r < n && comp(ind[ind_idx(r)], ind[ind_idx(index)]))
        {
            index = r;
            if(comp(ind[ind_idx(l)], ind[ind_idx(r)]))
            {
                index = l;
            }
        }

        if(index == pre_index)
        {
            break;
        }

        swap(ind[ind_idx(index)], ind[ind_idx(pre_index)]);
    }
}

template <class IndIndex, class Compare>
__device__ inline void build_heap(int64_t* ind,
                                  int n,
                                  IndIndex ind_idx,
                                  Compare comp)
{
    for(int j = n / 2 - 1; j >= 0; j--)
    {
        heap_heapify(ind, n, j, ind_idx, comp);
    }
}

template <class IndIndex, class Compare>
__device__ inline void heap_update(int64_t* ind,
                                int n,
                                int val,
                                IndIndex ind_idx,
                                Compare comp)
{
    if(comp(val, ind[ind_idx(0)]))
    {
        return;
    }

    ind[ind_idx(0)] = val;
    heap_heapify(ind, n, 0, ind_idx, comp);
}

template <class IndIndex, class Compare>
__device__ inline void heap_sort(int64_t* ind,
                                 int n,
                                 IndIndex ind_idx,
                                 Compare comp)
{
    build_heap(ind, n, ind_idx, comp);
    for(int j = n - 1; j > 0; --j)
    {
        swap(ind[ind_idx(0)], ind[ind_idx(j)]);
        heap_heapify(ind, j, 0, ind_idx, comp);
    }
}

template <class IndIndex, class Compare>
__device__ inline void topk_value(int64_t* const ind,
                                  int n,
                                  int k,
                                  IndIndex ind_idx,
                                  Compare comp)
{
    build_heap(ind, k, ind_idx, comp);
    for(int j = k; j < n; ++j)
    {
        heap_update(ind, k, j, ind_idx, comp);
    }
}


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
                    auto iidx = idx;
                    iidx[axis] = ii;
                    return iss.index(iidx);
                };

                auto out_idx = [&](int ii) {
                    auto iidx = idx;
                    iidx[axis] = ii;
                    return oss.index(iidx);
                };

                auto compare = [=](auto ii, auto jj)
                {
                    return largest ? std::less<>{}(data[in_idx(ii)], data[in_idx(jj)]) : std::greater<>{}(data[in_idx(ii)], data[in_idx(jj)]);
                };

                for(int j = 0; j < k; ++j)
                {
                    ind[out_idx(j)] = j;
                }

                topk_value(ind, axis_dim, k, out_idx, compare);
                heap_sort(ind, k, out_idx, compare);

                // read output
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
