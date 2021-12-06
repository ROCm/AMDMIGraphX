#ifndef MIGRAPHX_GUARD_KERNELS_GENERIC_CONSTANT_HPP
#define MIGRAPHX_GUARD_KERNELS_GENERIC_CONSTANT_HPP

namespace migraphx {

template <class F>
struct generic_constant
{
    static constexpr auto value = F{}();
    using value_type            = decltype(value);
    using type                  = generic_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

template <class F>
constexpr generic_constant<F> make_generic_constant(F)
{
    return {};
}

// NOLINTNEXTLINE
#define MIGRAPHX_MAKE_CONSTANT(x)                           \
    make_generic_constant([] {                              \
        struct fun                                          \
        {                                                   \
            constexpr auto operator()() const { return x; } \
        };                                                  \
        return fun{};                                       \
    }())

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_GENERIC_CONSTANT_HPP
