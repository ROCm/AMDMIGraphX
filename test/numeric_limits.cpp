#include <dual_test.hpp>
#include <rocm/limits.hpp>
#include <rocm/type_traits.hpp>

#define CHECK_NUMERIC_LIMITS_MEM(expected, ...) \
static_assert(rocm::is_same<expected, rocm::remove_cv_t<decltype(__VA_ARGS__)>>{})

template<class T, bool Specialized, class U=T>
constexpr void test_numeric_limits()
{
    using nl = rocm::numeric_limits<T>;
    static_assert(nl::is_specialized == Specialized);
    if constexpr(rocm::is_integral<U>{}) {
        static_assert(nl::is_integer);
        static_assert(nl::is_exact);
        static_assert(nl::is_signed == not rocm::is_unsigned<U>{});
    }
    if constexpr(rocm::is_floating_point<U>{}) {
        static_assert(not nl::is_integer);
        static_assert(not nl::is_exact);
        static_assert(nl::is_signed);
    }
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_specialized);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::digits);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::digits10);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::max_digits10);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_signed);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_integer);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_exact);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::radix);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::min_exponent);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::min_exponent10);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::max_exponent);
    CHECK_NUMERIC_LIMITS_MEM(int, nl::max_exponent10);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::has_infinity);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::has_quiet_NaN);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::has_signaling_NaN);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_iec559);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_bounded);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_modulo);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::traps);
    CHECK_NUMERIC_LIMITS_MEM(bool, nl::tinyness_before);
    CHECK_NUMERIC_LIMITS_MEM(rocm::float_round_style, nl::round_style);
    CHECK_NUMERIC_LIMITS_MEM(U, nl::min());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::lowest());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::max());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::epsilon());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::round_error());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::infinity());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::quiet_NaN());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::signaling_NaN());
    CHECK_NUMERIC_LIMITS_MEM(U, nl::denorm_min());
}

template<class T, bool Specialized=true>
constexpr void test_numeric_limits_all()
{
    test_numeric_limits<T, Specialized, T>();
    test_numeric_limits<const T, Specialized, T>();
    test_numeric_limits<volatile T, Specialized, T>();
    test_numeric_limits<const volatile T, Specialized, T>();
}

struct foo {};

DUAL_TEST_CASE()
{
    // test_numeric_limits_all<bool>();
    test_numeric_limits_all<char>();
    test_numeric_limits_all<signed char>();
    test_numeric_limits_all<unsigned char>();
    test_numeric_limits_all<wchar_t>();
    test_numeric_limits_all<char16_t>();
    test_numeric_limits_all<char32_t>();
    test_numeric_limits_all<short>();
    test_numeric_limits_all<unsigned short>();
    test_numeric_limits_all<int>();
    test_numeric_limits_all<unsigned int>();
    test_numeric_limits_all<long>();
    test_numeric_limits_all<unsigned long>();
    test_numeric_limits_all<long long>();
    test_numeric_limits_all<unsigned long long>();
    test_numeric_limits_all<double>();
    test_numeric_limits_all<float>();
#ifdef __FLT16_MAX__
    test_numeric_limits_all<_Float16>();
#endif
    test_numeric_limits_all<foo, false>();
}
