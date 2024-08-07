#ifndef MIGRAPHX_GUARD_KERNELS_PERMUTATION_HPP
#define MIGRAPHX_GUARD_KERNELS_PERMUTATION_HPP

#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/tuple.hpp>

namespace migraphx {

template <class Array1, class Array2>
constexpr auto reorder_dims(const Array1& dims, const Array2& permutation)
{
    return generate_array<typename Array1::value_type>(
        dims.size(), [&](auto i) { return dims[permutation[i]]; });
}

template <class T, T... Xs, class U, U... Ys>
constexpr auto reorder_dims(integral_const_array<T, Xs...>, integral_const_array<U, Ys...>)
{
    return return_array_c([] { 
        constexpr integral_const_array<T, Xs...> dims{};
        constexpr integral_const_array<U, Ys...> permutation{};
        return reorder_dims(dims.base(), permutation.base()); 
    });
}

template <class Array>
constexpr auto invert_permutation(const Array& permutation)
{
    return reorder_dims(permutation, permutation);
}

template <class Shape>
constexpr auto find_permutation(Shape)
{
    return return_array_c([] {
        constexpr Shape s{};
        typename Shape::index_array perm;
        iota(perm.begin(), perm.end(), 0);
        sort(perm.begin(), perm.end(), by([&](auto x) {
                 return make_tuple(s.strides[x], s.lens[x]);
             }, greater{}));
        return perm;
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PERMUTATION_HPP
