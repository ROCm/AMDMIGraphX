#include <rocm/functional/operations.hpp>
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

template <class T, class = void>
struct has_is_transparent : rocm::bool_constant<false>
{
};

template <class T>
struct has_is_transparent<T, typename T::is_transparent> : rocm::bool_constant<true>
{
};

// Arithmetic binary operations

TEST_CASE(plus_typed)
{
    constexpr auto op = rocm::plus<int>{};
    EXPECT(op(3, 5) == 8);
    EXPECT(op(-3, 3) == 0);
    EXPECT(op(0, 0) == 0);
    EXPECT(op(-2, -3) == -5);
    static_assert(rocm::plus<int>{}(10, 20) == 30);
}

TEST_CASE(minus_typed)
{
    constexpr auto op = rocm::minus<int>{};
    EXPECT(op(5, 3) == 2);
    EXPECT(op(3, 5) == -2);
    EXPECT(op(0, 0) == 0);
    EXPECT(op(-2, -3) == 1);
    static_assert(rocm::minus<int>{}(10, 3) == 7);
}

TEST_CASE(multiplies_typed)
{
    constexpr auto op = rocm::multiplies<int>{};
    EXPECT(op(3, 5) == 15);
    EXPECT(op(-3, 5) == -15);
    EXPECT(op(0, 100) == 0);
    EXPECT(op(1, 42) == 42);
    static_assert(rocm::multiplies<int>{}(6, 7) == 42);
}

TEST_CASE(divides_typed)
{
    constexpr auto op = rocm::divides<int>{};
    EXPECT(op(10, 2) == 5);
    EXPECT(op(7, 2) == 3);
    EXPECT(op(-10, 2) == -5);
    EXPECT(op(0, 5) == 0);
    static_assert(rocm::divides<int>{}(20, 4) == 5);
}

TEST_CASE(modulus_typed)
{
    constexpr auto op = rocm::modulus<int>{};
    EXPECT(op(10, 3) == 1);
    EXPECT(op(9, 3) == 0);
    EXPECT(op(7, 2) == 1);
    EXPECT(op(0, 5) == 0);
    static_assert(rocm::modulus<int>{}(10, 3) == 1);
}

// Bitwise binary operations

TEST_CASE(bit_and_typed)
{
    constexpr auto op = rocm::bit_and<int>{};
    EXPECT(op(0b1100, 0b1010) == 0b1000);
    EXPECT(op(0xFF, 0x0F) == 0x0F);
    EXPECT(op(0, 0xFF) == 0);
    EXPECT(op(0xFF, 0xFF) == 0xFF);
}

TEST_CASE(bit_or_typed)
{
    constexpr auto op = rocm::bit_or<int>{};
    EXPECT(op(0b1100, 0b1010) == 0b1110);
    EXPECT(op(0xFF, 0x0F) == 0xFF);
    EXPECT(op(0, 0xFF) == 0xFF);
    EXPECT(op(0, 0) == 0);
}

TEST_CASE(bit_xor_typed)
{
    constexpr auto op = rocm::bit_xor<int>{};
    EXPECT(op(0b1100, 0b1010) == (0b1100 ^ 0b1010));
    EXPECT(op(0xFF, 0x0F) == (0xFF ^ 0x0F));
    EXPECT(op(0xFF, 0xFF) == 0);
    EXPECT(op(0, 0) == 0);
}

// Comparison operations

TEST_CASE(equal_to_typed)
{
    constexpr auto op = rocm::equal_to<int>{};
    EXPECT(op(5, 5));
    EXPECT(not op(5, 3));
    EXPECT(op(0, 0));
    EXPECT(op(-1, -1));
    static_assert(rocm::equal_to<int>{}(5, 5));
}

TEST_CASE(not_equal_to_typed)
{
    constexpr auto op = rocm::not_equal_to<int>{};
    EXPECT(op(5, 3));
    EXPECT(not op(5, 5));
    EXPECT(not op(0, 0));
    EXPECT(op(-1, 1));
}

TEST_CASE(greater_typed)
{
    constexpr auto op = rocm::greater<int>{};
    EXPECT(op(5, 3));
    EXPECT(not op(3, 5));
    EXPECT(not op(5, 5));
    EXPECT(op(0, -1));
}

TEST_CASE(less_typed)
{
    constexpr auto op = rocm::less<int>{};
    EXPECT(op(3, 5));
    EXPECT(not op(5, 3));
    EXPECT(not op(5, 5));
    EXPECT(op(-1, 0));
}

TEST_CASE(greater_equal_typed)
{
    constexpr auto op = rocm::greater_equal<int>{};
    EXPECT(op(5, 3));
    EXPECT(op(5, 5));
    EXPECT(not op(3, 5));
    EXPECT(op(0, 0));
}

TEST_CASE(less_equal_typed)
{
    constexpr auto op = rocm::less_equal<int>{};
    EXPECT(op(3, 5));
    EXPECT(op(5, 5));
    EXPECT(not op(5, 3));
    EXPECT(op(0, 0));
}

// Logical binary operations

TEST_CASE(logical_and_typed)
{
    constexpr auto op = rocm::logical_and<bool>{};
    EXPECT(op(true, true));
    EXPECT(not op(true, false));
    EXPECT(not op(false, true));
    EXPECT(not op(false, false));
}

TEST_CASE(logical_or_typed)
{
    constexpr auto op = rocm::logical_or<bool>{};
    EXPECT(op(true, true));
    EXPECT(op(true, false));
    EXPECT(op(false, true));
    EXPECT(not op(false, false));
}

// Unary operations

TEST_CASE(negate_typed)
{
    constexpr auto op = rocm::negate<int>{};
    EXPECT(op(5) == -5);
    EXPECT(op(-5) == 5);
    EXPECT(op(0) == 0);
    static_assert(rocm::negate<int>{}(5) == -5);
}

TEST_CASE(logical_not_typed)
{
    constexpr auto op = rocm::logical_not<bool>{};
    EXPECT(op(false));
    EXPECT(not op(true));
}

TEST_CASE(bit_not_typed)
{
    constexpr auto op = rocm::bit_not<int>{};
    EXPECT(op(0) == ~0);
    EXPECT(op(-1) == ~(-1));
    EXPECT(op(0b1100) == ~0b1100);
}

// Return type checks for typed specializations

TEST_CASE(arithmetic_return_types)
{
    EXPECT(rocm::is_same<decltype(rocm::plus<int>{}(1, 2)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::minus<int>{}(1, 2)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::multiplies<int>{}(1, 2)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::divides<int>{}(1, 2)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::modulus<int>{}(1, 2)), int>{});
}

TEST_CASE(comparison_return_types)
{
    EXPECT(rocm::is_same<decltype(rocm::equal_to<int>{}(1, 2)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::not_equal_to<int>{}(1, 2)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::greater<int>{}(1, 2)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::less<int>{}(1, 2)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::greater_equal<int>{}(1, 2)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::less_equal<int>{}(1, 2)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::logical_and<bool>{}(true, false)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::logical_or<bool>{}(true, false)), bool>{});
}

TEST_CASE(unary_return_types)
{
    EXPECT(rocm::is_same<decltype(rocm::negate<int>{}(1)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::logical_not<bool>{}(true)), bool>{});
    EXPECT(rocm::is_same<decltype(rocm::bit_not<int>{}(1)), int>{});
}

// Transparent specialization (is_transparent)

TEST_CASE(binary_arithmetic_is_transparent)
{
    EXPECT(has_is_transparent<rocm::plus<>>{});
    EXPECT(has_is_transparent<rocm::minus<>>{});
    EXPECT(has_is_transparent<rocm::multiplies<>>{});
    EXPECT(has_is_transparent<rocm::divides<>>{});
    EXPECT(has_is_transparent<rocm::modulus<>>{});
}

TEST_CASE(binary_bitwise_is_transparent)
{
    EXPECT(has_is_transparent<rocm::bit_and<>>{});
    EXPECT(has_is_transparent<rocm::bit_or<>>{});
    EXPECT(has_is_transparent<rocm::bit_xor<>>{});
}

TEST_CASE(binary_comparison_is_transparent)
{
    EXPECT(has_is_transparent<rocm::equal_to<>>{});
    EXPECT(has_is_transparent<rocm::not_equal_to<>>{});
    EXPECT(has_is_transparent<rocm::greater<>>{});
    EXPECT(has_is_transparent<rocm::less<>>{});
    EXPECT(has_is_transparent<rocm::greater_equal<>>{});
    EXPECT(has_is_transparent<rocm::less_equal<>>{});
}

TEST_CASE(binary_logical_is_transparent)
{
    EXPECT(has_is_transparent<rocm::logical_and<>>{});
    EXPECT(has_is_transparent<rocm::logical_or<>>{});
}

TEST_CASE(unary_is_transparent)
{
    EXPECT(has_is_transparent<rocm::negate<>>{});
    EXPECT(has_is_transparent<rocm::logical_not<>>{});
    EXPECT(has_is_transparent<rocm::bit_not<>>{});
}

TEST_CASE(typed_not_transparent)
{
    EXPECT(not has_is_transparent<rocm::plus<int>>{});
    EXPECT(not has_is_transparent<rocm::minus<int>>{});
    EXPECT(not has_is_transparent<rocm::equal_to<int>>{});
    EXPECT(not has_is_transparent<rocm::negate<int>>{});
    EXPECT(not has_is_transparent<rocm::logical_not<bool>>{});
}

// Default template argument resolves to void

TEST_CASE(default_template_arg)
{
    EXPECT(rocm::is_same<rocm::plus<>, rocm::plus<void>>{});
    EXPECT(rocm::is_same<rocm::minus<>, rocm::minus<void>>{});
    EXPECT(rocm::is_same<rocm::multiplies<>, rocm::multiplies<void>>{});
    EXPECT(rocm::is_same<rocm::negate<>, rocm::negate<void>>{});
    EXPECT(rocm::is_same<rocm::logical_not<>, rocm::logical_not<void>>{});
}

// Bitwise operations return type for typed specializations

TEST_CASE(bitwise_return_types)
{
    EXPECT(rocm::is_same<decltype(rocm::bit_and<int>{}(1, 2)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::bit_or<int>{}(1, 2)), int>{});
    EXPECT(rocm::is_same<decltype(rocm::bit_xor<int>{}(1, 2)), int>{});
}

// Operations with long type

TEST_CASE(plus_long)
{
    constexpr auto op = rocm::plus<long>{};
    EXPECT(op(100000L, 200000L) == 300000L);
    EXPECT(op(-100000L, 100000L) == 0L);
}

TEST_CASE(multiplies_long)
{
    constexpr auto op = rocm::multiplies<long>{};
    EXPECT(op(10000L, 10000L) == 100000000L);
    EXPECT(op(-1L, 42L) == -42L);
}
