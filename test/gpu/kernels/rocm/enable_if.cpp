#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

template <class T>
struct check : rocm::bool_constant<false>
{
};

template <>
struct check<long> : rocm::bool_constant<true>
{
};

struct construct
{
    template <class T>
    constexpr construct(T, typename rocm::enable_if<check<T>{}>::type* = nullptr) : v(true)
    {
    }
    template <class T>
    constexpr construct(T, typename rocm::enable_if<not check<T>{}>::type* = nullptr) : v(false)
    {
    }
    constexpr bool value() const { return v; }

    private:
    bool v;
};

template <class T, class E = void>
struct specialize;

template <class T>
struct specialize<T, typename rocm::enable_if<check<T>{}>::type> : rocm::bool_constant<true>
{
};

template <class T>
struct specialize<T, typename rocm::enable_if<not check<T>{}>::type> : rocm::bool_constant<false>
{
};

template <class T>
constexpr typename rocm::enable_if<check<T>{}, bool>::type returns(T)
{
    return true;
}

template <class T>
constexpr typename rocm::enable_if<not check<T>{}, bool>::type returns(T)
{
    return false;
}

template <class T>
constexpr rocm::enable_if_t<check<T>{}, bool> alias(T)
{
    return true;
}

template <class T>
constexpr rocm::enable_if_t<not check<T>{}, bool> alias(T)
{
    return false;
}

TEST_CASE(test)
{
    EXPECT(not construct(1).value());
    EXPECT(construct(1L).value());
    EXPECT(not specialize<int>{});
    EXPECT(specialize<long>{});
    EXPECT(not returns(1));
    EXPECT(returns(1L));
    EXPECT(not alias(1));
    EXPECT(alias(1L));
}
