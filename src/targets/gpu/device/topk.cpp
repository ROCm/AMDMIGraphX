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

struct greater
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return (x > y);
    }
};

struct less
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return (x < y);
    }
};

template <class T>
__device__ void swap(T& v1, T& v2)
{
    T v = v1;
    v1  = v2;
    v2  = v;
}

template <class T, class Op>
__device__ void heap_heapify(T* arr, int* idx, int n, int i, Op op)
{
    int index = i;
    while(index < n)
    {
        auto pre_index = index;
        int l          = 2 * index + 1;
        int r          = 2 * index + 2;

        if(l < n && op(arr[idx[l]], arr[idx[index]]))
        {
            index = l;
        }

        if(r < n && op(arr[idx[r]], arr[idx[index]]))
        {
            index = r;
            if(op(arr[idx[l]], arr[idx[r]]))
            {
                index = l;
            }
        }

        if(index == pre_index)
        {
            break;
        }
        swap(idx[index], idx[pre_index]);
    }
}

template <class T, class Op>
__device__ void build_heap(T* arr, int* idx, int n, Op op)
{
    for(int i = n / 2 - 1; i >= 0; i--)
    {
        heap_heapify(arr, idx, n, i, op);
    }
}

template <class T, class Op>
__device__ void heap_add(T* arr, int* idx, int n, const int& val, Op op)
{
    if(op(arr[val], arr[idx[0]]))
    {
        return;
    }

    idx[0] = val;
    heap_heapify(arr, idx, n, 0, op);
}

template <class T, class Op>
__device__ void heapSort(T* arr, int* idx, int n, Op op)
{
    build_heap(arr, idx, n, op);

    for(int i = n - 1; i > 0; i--)
    {
        swap(idx[0], idx[i]);
        heap_heapify(arr, idx, i, 0, op);
    }
}

template <class T, class Op>
__device__ void topk_value(T* arr, int* idx, int n, int k, Op op)
{
    build_heap(arr, idx, k, op);
    for(int i = k; i < n; ++i)
    {
        heap_add(arr, idx, k, idx[i], op);
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
            using type = typename decltype(input)::value_type;
            auto* data = device_cast(input.data());
            auto* out  = device_cast(out_val.data());
            ind_res.visit([&](auto out_ind) {
                auto* ind = device_cast(out_ind.data());
                gs_launch(stream, elem_num, 256)([&](auto i) __device__ {
                    auto idx   = css.multi(i);
                    auto idata = iss.index(idx);
                    // for (int j = 0; j < axis_dim; ++j)
                    // {
                    //     sidx[j] = j;
                    //     idx[axis] = j;
                    //     val[j] = data[iss.index(idx)];
                    // }
                    // topk_value(val, sidx, axis_dim, k, greater{});
                });
            });
        });

    return argument({val_res, ind_res});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
