#include "migraphx/gpu/device/shape.hpp"
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/topk.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/visit.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
__device__ void swap(T& v1, T& v2)
{
    T v = v1;
    v1  = v2;
    v2  = v;
}

template <class T, index_int N, class Op>
__device__ inline void heap_heapify(T* arr,
                                    int64_t* ind,
                                    const int64_t i,
                                    const hip_shape<N>& oss,
                                    const hip_shape<N>& iss,
                                    const hip_shape<N>& css,
                                    int n,
                                    int index,
                                    const int64_t axis,
                                    Op op)
{
    auto idx  = css.multi(i);
    auto idxl = idx;
    auto idxr = idx;
    auto idxp = idx;
    while(index < n)
    {
        auto pre_index = index;
        int l          = 2 * index + 1;
        int r          = 2 * index + 2;

        idx[axis]  = index;
        idx[axis]  = ind[oss.index(idx)];
        idxl[axis] = l;
        idxl[axis] = ind[oss.index(idxl)];
        idxr[axis] = r;
        idxr[axis] = ind[oss.index(idxr)];

        if(l < n && op(arr[iss.index(idxl)], arr[iss.index(idx)]))
        {
            index = l;
        }

        if(r < n && op(arr[iss.index(idxr)], arr[iss.index(idx)]))
        {
            index = r;
            if(op(arr[iss.index(idxl)], arr[iss.index(idxr)]))
            {
                index = l;
            }
        }

        if(index == pre_index)
        {
            break;
        }

        idx[axis]  = index;
        idxp[axis] = pre_index;
        swap(ind[oss.index(idx)], ind[oss.index(idxp)]);
    }
}

template <class T, index_int N, class Op>
__device__ inline void build_heap(T* arr,
                                  int64_t* ind,
                                  const int64_t i,
                                  const hip_shape<N>& oss,
                                  const hip_shape<N>& iss,
                                  const hip_shape<N>& css,
                                  int n,
                                  const int64_t axis,
                                  Op op)
{
    for(int j = n / 2 - 1; j >= 0; j--)
    {
        heap_heapify(arr, ind, i, oss, iss, css, n, j, axis, op);
    }
}

template <class T, index_int N, class Op>
__device__ inline void heap_add(T* arr,
                                int64_t* ind,
                                const int64_t i,
                                const hip_shape<N>& oss,
                                const hip_shape<N>& iss,
                                const hip_shape<N>& css,
                                int n,
                                int val,
                                const int64_t axis,
                                Op op)
{
    auto idx   = css.multi(i);
    auto vidx  = idx;
    idx[axis]  = ind[oss.index(idx)];
    vidx[axis] = val;

    if(op(arr[iss.index(vidx)], arr[iss.index(idx)]))
    {
        return;
    }

    idx                 = css.multi(i);
    ind[oss.index(idx)] = val;
    heap_heapify(arr, ind, i, oss, iss, css, n, 0, axis, op);
}

template <class T, index_int N, class Op>
__device__ inline void heap_sort(T* arr,
                                 int64_t* ind,
                                 const int64_t i,
                                 const hip_shape<N>& oss,
                                 const hip_shape<N>& iss,
                                 const hip_shape<N>& css,
                                 int n,
                                 const int64_t axis,
                                 Op op)
{
    build_heap(arr, ind, i, oss, iss, css, n, axis, op);
    auto idx  = css.multi(i);
    auto idxj = idx;
    for(int j = n - 1; j > 0; --j)
    {
        idxj[axis] = j;
        swap(ind[oss.index(idx)], ind[oss.index(idxj)]);
        heap_heapify(arr, ind, i, oss, iss, css, j, 0, axis, op);
    }
}

template <class T, index_int N, class Op>
__device__ inline void topk_value(const T* arr,
                                  int64_t* ind,
                                  const int64_t i,
                                  const hip_shape<N>& oss,
                                  const hip_shape<N>& iss,
                                  const hip_shape<N>& css,
                                  int n,
                                  int k,
                                  const int64_t axis,
                                  Op op)
{
    build_heap(arr, ind, i, oss, iss, css, k, axis, op);
    for(int j = k; j < n; ++j)
    {
        heap_add(arr, ind, i, oss, iss, css, k, j, axis, op);
    }
}

argument topk(hipStream_t stream,
              argument val_res,
              argument ind_res,
              argument arg,
              int64_t k,
              int64_t axis,
              bool sorted,
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
            auto* data = device_cast(input.data());
            auto* out  = device_cast(out_val.data());
            auto* ind  = ind_res.cast<int64_t>();
            gs_launch(stream, elem_num)([=](auto i) __device__ {
                auto idx = css.multi(i);
                for(int j = 0; j < k; ++j)
                {
                    idx[axis]           = j;
                    ind[oss.index(idx)] = j;
                }

                largest ? topk_value(data, ind, i, oss, iss, css, axis_dim, k, axis, std::less<>{}) : topk_value(data, ind, i, oss, iss, css, axis_dim, k, axis, std::greater<>{});

                // if outputs are sorted, sort them
                if(sorted)
                {
                    largest ? heap_sort(data, ind, i, oss, iss, css, k, axis, std::less<>{}) : heap_sort(data, ind, i, oss, iss, css, k, axis, std::greater<>{});
                }

                // read output
                idx = css.multi(i);
                for(int j = 0; j < k; ++j)
                {
                    auto in_idx         = idx;
                    idx[axis]           = j;
                    in_idx[axis]        = ind[oss.index(idx)];
                    out[oss.index(idx)] = data[iss.index(in_idx)];
                }
            });
        });

    return argument({val_res, ind_res});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
