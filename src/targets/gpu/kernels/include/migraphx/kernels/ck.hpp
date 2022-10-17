#ifndef MIGRAPHX_GUARD_KERNELS_CK_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_HPP

#include <migraphx/kernels/debug.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <ck/utility/common_header.hpp>
#include <ck/tensor_description/tensor_descriptor.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>

namespace migraphx {

namespace detail {
template <class T>
struct to_ck_type_impl
{
    using type = T;
};
template <>
struct to_ck_type_impl<migraphx::half>
{
    using type = ck::half_t;
};

template <class Shape>
constexpr bool is_row_major()
{
    constexpr auto strides = Shape{}.strides;
    MIGRAPHX_ASSERT(strides.size() >= 2);
    if(strides.back() == 1)
    {
        MIGRAPHX_ASSERT(not Shape{}.is_transposed());
        return true;
    }
    MIGRAPHX_ASSERT(strides[strides.size() - 2] == 1);

    return false;
}

} // namespace detail

template <class T>
using to_ck_type = typename detail::to_ck_type_impl<T>::type;

template <class Shape>
using to_ck_gemm_layout = conditional_t<detail::is_row_major<get_shape_c<Shape>>(),
                                        ck::tensor_layout::gemm::RowMajor,
                                        ck::tensor_layout::gemm::ColumnMajor>;

template <class Tensor>
constexpr auto to_ck_tensor()
{
    constexpr auto s = get_shape_c<Tensor>{};
    return sequence(s.lens.size(), [&](auto... is) {
        return ck::make_naive_tensor_descriptor(ck::make_tuple(s.lens[is]...),
                                                ck::make_tuple(s.strides[is]...));
    });
}

template <class F>
struct ck_function_adaptor : F
{
    template <class... Ts>
    constexpr ck_function_adaptor(Ts&&... xs) : F(static_cast<Ts&&>(xs)...)
    {
    }

    template <class T, class... Ts>
    constexpr void operator()(T& out, Ts&&... xs) const
    {
        out = static_cast<const F&>(*this)(static_cast<Ts&&>(xs)...);
    }
};

struct ck_nop
{
    template <class T>
    constexpr void operator()(T&) const
    {
    }
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CK_HPP
