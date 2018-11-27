#ifndef MIGRAPHX_GUARD_MIGRAPHX_MANAGE_PTR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_MANAGE_PTR_HPP

#include <memory>
#include <type_traits>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F, F f> // NOLINT
struct manage_deleter
{
    template <class T>
    void operator()(T* x) const
    {
        if(x != nullptr)
        {
            f(x);
        }
    }
};

struct null_deleter
{
    template <class T>
    void operator()(T*) const
    {
    }
};

template <class T, class F, F f> // NOLINT
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

template <class T>
struct element_type
{
    using type = typename T::element_type;
};

template <class T>
using remove_ptr = typename std::
    conditional_t<std::is_pointer<T>{}, std::remove_pointer<T>, element_type<T>>::type;

template <class T>
using shared = std::shared_ptr<remove_ptr<T>>;

template <class T>
shared<T> share(T p)
{
    return shared<T>{std::move(p)};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#define MIGRAPHX_MANAGE_PTR(T, F) \
    migraphx::manage_ptr<std::remove_pointer_t<T>, decltype(&F), &F> // NOLINT

#endif
