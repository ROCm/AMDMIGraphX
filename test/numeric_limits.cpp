#include <dual_test.hpp>
#include <rocm/limits.hpp>
#include <rocm/type_traits.hpp>

#define ROCM_CHECK_NUMERIC_LIMITS_MEM(expected, ...) \
    static_assert(rocm::is_same<expected, rocm::remove_cv_t<decltype(__VA_ARGS__)>>{})

template <class T, bool Specialized, class U = T>
constexpr void test_numeric_limits()
{
    using nl = rocm::numeric_limits<T>;
    static_assert(nl::is_specialized == Specialized);
    if constexpr(rocm::is_integral<U>{})
    {
        static_assert(nl::is_integer);
        static_assert(nl::is_exact);
        static_assert(nl::is_signed == not rocm::is_unsigned<U>{});
    }
    if constexpr(rocm::is_floating_point<U>{})
    {
        static_assert(not nl::is_integer);
        static_assert(not nl::is_exact);
        static_assert(nl::is_signed);
    }
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_specialized);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::digits);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::digits10);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::max_digits10);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_signed);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_integer);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_exact);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::radix);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::min_exponent);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::min_exponent10);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::max_exponent);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(int, nl::max_exponent10);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::has_infinity);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::has_quiet_NaN);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::has_signaling_NaN);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_iec559);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_bounded);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::is_modulo);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::traps);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(bool, nl::tinyness_before);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(rocm::float_round_style, nl::round_style);
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::min());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::lowest());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::max());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::epsilon());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::round_error());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::infinity());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::quiet_NaN());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::signaling_NaN());
    ROCM_CHECK_NUMERIC_LIMITS_MEM(U, nl::denorm_min());
}

template <class T, bool Specialized = true>
constexpr void test_numeric_limits_all()
{
    test_numeric_limits<T, Specialized, T>();
    test_numeric_limits<const T, Specialized, T>();
    test_numeric_limits<volatile T, Specialized, T>();
    test_numeric_limits<const volatile T, Specialized, T>();
}

struct foo
{
};

ROCM_DUAL_TEST_CASE()
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
