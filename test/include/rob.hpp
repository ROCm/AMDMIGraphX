#ifndef MIGRAPHX_GUARD_ROB_HPP
#define MIGRAPHX_GUARD_ROB_HPP

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

// Used to access private member variables
template <class Tag>
struct stowed
{
    static typename Tag::type value;
};
template <class Tag>
typename Tag::type stowed<Tag>::value;

template <class Tag, typename Tag::type X>
struct stow_private
{
    stow_private() noexcept { stowed<Tag>::value = X; }
    static stow_private instance;
};
template <class Tag, typename Tag::type X>
stow_private<Tag, X> stow_private<Tag, X>::instance;

template <class C, class T>
struct mem_data_ptr
{
    using type = T C::*;
};

// NOLINTNEXTLINE
#define MIGRAPHX_ROB(name, Type, C, mem)               \
    struct name##_tag : mem_data_ptr<C, Type>          \
    {                                                  \
    };                                                 \
    template struct stow_private<name##_tag, &C::mem>; \
    template <class T>                                 \
    auto& name(T&& x)                                  \
    {                                                  \
        return x.*stowed<name##_tag>::value;           \
    }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
