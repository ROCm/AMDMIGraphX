#ifndef MIGRAPHX_GUARD_KERNELS_COPY_HPP
#define MIGRAPHX_GUARD_KERNELS_COPY_HPP

#include <migraphx/kernels/print.hpp>
#include <migraphx/kernels/vectorize.hpp>

namespace migraphx {

template<class Index, class T, class U, class Size>
__device__ void local_vector_copy(Index idx, T* src, U* dst, Size size)
{
    constexpr auto n = find_vectorize_size([&](auto i) { return (size % i) == 0; });
    auto vsrc           = as_vec<n>(remove_bool(src));
    auto vdst       = as_vec<n>(remove_bool(dst));
    index_int vsize = size / n;
    idx.local_stride(vsize, [&](auto i) { vdst[i] = vsrc[i]; });
}

template<class Index, class T, class U>
__device__ void local_tensor_copy(Index idx, T src, U dst)
{
    constexpr auto src_shape = get_shape_c<T>{};
    constexpr auto dst_shape = get_shape_c<U>{};
    if constexpr(src_shape == dst_shape and (src_shape.packed() or src_shape.broadcasted()))
    {
        local_vector_copy(idx, src.data(), dst.data(), src_shape.element_space());
    }
    else
    {
        constexpr auto perm = find_permutation(src_shape, dst_shape);
        auto new_src = reorder_tensor_view(src, perm);
        auto new_dst = reorder_tensor_view(dst, perm);
        // println_once("new_src: ", new_src.get_shape());
        // println_once("new_dst: ", new_dst.get_shape());
        auto_vectorize()(new_src, new_dst)([&](auto vsrc, auto vdst) {
            index_int size = vsrc.get_shape().elements();
            idx.local_stride(size, [&](auto i) { vdst[i] = vsrc[i]; });
        });
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_COPY_HPP
