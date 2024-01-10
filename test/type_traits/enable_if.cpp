#include <type_traits_test.hpp>


template<class T>
struct check
    : rocm::bool_constant<false> { };

template<>
struct check<long>
    : rocm::bool_constant<true> { };

struct construct {
    template<class T>
    constexpr construct(T, typename rocm::enable_if<check<T>{}>::type* = nullptr)
        : v(true) { }
    template<class T>
    constexpr construct(T, typename rocm::enable_if<not check<T>{}>::type* = nullptr)
        : v(false) { }
    constexpr bool value() const {
        return v;
    }
private:
    bool v;
};

template<class T, class E = void>
struct specialize;

template<class T>
struct specialize<T, typename rocm::enable_if<check<T>{}>::type>
    : rocm::bool_constant<true> { };

template<class T>
struct specialize<T, typename rocm::enable_if<not check<T>{}>::type>
    : rocm::bool_constant<false> { };

template<class T>
constexpr typename rocm::enable_if<check<T>{}, bool>::type returns(T)
{
    return true;
}

template<class T>
constexpr typename rocm::enable_if<not check<T>{}, bool>::type returns(T)
{
    return false;
}

template<class T>
constexpr rocm::enable_if_t<check<T>{}, bool> alias(T)
{
    return true;
}

template<class T>
constexpr rocm::enable_if_t<not check<T>{}, bool> alias(T)
{
    return false;
}

ROCM_DUAL_TEST_CASE()
{
    static_assert(not construct(1).value());
    static_assert(construct(1L).value());
    static_assert(not specialize<int>{});
    static_assert(specialize<long>{});
    static_assert(not returns(1));
    static_assert(returns(1L));
    static_assert(not alias(1));
    static_assert(alias(1L));
}

