#include <type_traits_test.hpp>

struct test_operators_with_value
{
    static_assert(rocm::true_type::value);
    static_assert(not rocm::false_type::value);

    typedef rocm::integral_constant<int, 1> one_type;
    typedef rocm::integral_constant<int, 2> two_type;

    typedef decltype(not rocm::true_type()) not_rocm::true_type;
    static_assert(not_rocm::true_type::value == false);

    typedef decltype(rocm::true_type() && rocm::false_type()) rocm::true_type_and_rocm::false_type;
    static_assert(rocm::true_type_and_rocm::false_type::value == false);
    typedef decltype(rocm::true_type() || rocm::false_type()) rocm::true_type_or_rocm::false_type;
    static_assert(rocm::true_type_or_rocm::false_type::value == true);

    typedef decltype(not two_type()) not_two_type;
    static_assert(not_two_type::value == (not 2));

// GCC confuses the complement operator with the destructor
#ifdef __clang__
    typedef decltype(~two_type()) compl_two_type;
    static_assert(compl_two_type::value == (~2));
#endif

    typedef decltype(+two_type()) unary_plus_two_type;
    static_assert(unary_plus_two_type::value == (+2));
    typedef decltype(-two_type()) unary_subtract_two_type;
    static_assert(unary_subtract_two_type::value == (-2));

    typedef decltype(one_type() + two_type()) one_type_add_two_type;
    static_assert(one_type_add_two_type::value == (1 + 2));
    typedef decltype(one_type() - two_type()) one_type_subtract_two_type;
    static_assert(one_type_subtract_two_type::value == (1 - 2));
    typedef decltype(one_type() * two_type()) one_type_multiply_two_type;
    static_assert(one_type_multiply_two_type::value == (1 * 2));
    typedef decltype(one_type() / two_type()) one_type_divide_two_type;
    static_assert(one_type_divide_two_type::value == (1 / 2));
    typedef decltype(one_type() % two_type()) one_type_remainder_two_type;
    static_assert(one_type_remainder_two_type::value == (1 % 2));
    typedef decltype(one_type() >> two_type()) one_type_shift_right_two_type;
    static_assert(one_type_shift_right_two_type::value == (1 >> 2));
    typedef decltype(one_type() << two_type()) one_type_shift_left_two_type;
    static_assert(one_type_shift_left_two_type::value == (1 << 2));
    typedef decltype(one_type() > two_type()) one_type_greater_than_two_type;
    static_assert(one_type_greater_than_two_type::value == (1 > 2));
    typedef decltype(one_type() < two_type()) one_type_less_than_two_type;
    static_assert(one_type_less_than_two_type::value == (1 < 2));
    typedef decltype(one_type() <= two_type()) one_type_less_than_equal_two_type;
    static_assert(one_type_less_than_equal_two_type::value == (1 <= 2));
    typedef decltype(one_type() >= two_type()) one_type_greater_than_equal_two_type;
    static_assert(one_type_greater_than_equal_two_type::value == (1 >= 2));
    typedef decltype(one_type() == two_type()) one_type_equal_two_type;
    static_assert(one_type_equal_two_type::value == (1 == 2));
    typedef decltype(one_type() != two_type()) one_type_not_equal_two_type;
    static_assert(one_type_not_equal_two_type::value == (1 != 2));
    typedef decltype(one_type() & two_type()) one_type_bit_and_two_type;
    static_assert(one_type_bit_and_two_type::value == (1 & 2));
    typedef decltype(one_type() ^ two_type()) one_type_xor_two_type;
    static_assert(one_type_xor_two_type::value == (1 ^ 2));
    typedef decltype(one_type() | two_type()) one_type_bit_or_two_type;
    static_assert(one_type_bit_or_two_type::value == (1 | 2));

};

struct test_operators_implicit_conversion
{
    static_assert(rocm::true_type());
    static_assert(not rocm::false_type());

    typedef rocm::integral_constant<int, 1> one_type;
    typedef rocm::integral_constant<int, 2> two_type;

    typedef decltype(not rocm::true_type()) not_rocm::true_type;
    static_assert(not_rocm::true_type() == false);

    typedef decltype(rocm::true_type() && rocm::false_type()) rocm::true_type_and_rocm::false_type;
    static_assert(rocm::true_type_and_rocm::false_type() == false);
    typedef decltype(rocm::true_type() || rocm::false_type()) rocm::true_type_or_rocm::false_type;
    static_assert(rocm::true_type_or_rocm::false_type() == true);

    typedef decltype(not two_type()) not_two_type;
    static_assert(not_two_type() == (not 2));

// GCC confuses the complement operator with the destructor
#ifdef __clang__
    typedef decltype(~two_type()) compl_two_type;
    static_assert(compl_two_type() == (~2));
#endif

    typedef decltype(+two_type()) unary_plus_two_type;
    static_assert(unary_plus_two_type() == (+2));
    typedef decltype(-two_type()) unary_subtract_two_type;
    static_assert(unary_subtract_two_type() == (-2));

    typedef decltype(one_type() + two_type()) one_type_add_two_type;
    static_assert(one_type_add_two_type() == (1 + 2));
    typedef decltype(one_type() - two_type()) one_type_subtract_two_type;
    static_assert(one_type_subtract_two_type() == (1 - 2));
    typedef decltype(one_type() * two_type()) one_type_multiply_two_type;
    static_assert(one_type_multiply_two_type() == (1 * 2));
    typedef decltype(one_type() / two_type()) one_type_divide_two_type;
    static_assert(one_type_divide_two_type() == (1 / 2));
    typedef decltype(one_type() % two_type()) one_type_remainder_two_type;
    static_assert(one_type_remainder_two_type() == (1 % 2));
    typedef decltype(one_type() >> two_type()) one_type_shift_right_two_type;
    static_assert(one_type_shift_right_two_type() == (1 >> 2));
    typedef decltype(one_type() << two_type()) one_type_shift_left_two_type;
    static_assert(one_type_shift_left_two_type() == (1 << 2));
    typedef decltype(one_type() > two_type()) one_type_greater_than_two_type;
    static_assert(one_type_greater_than_two_type() == (1 > 2));
    typedef decltype(one_type() < two_type()) one_type_less_than_two_type;
    static_assert(one_type_less_than_two_type() == (1 < 2));
    typedef decltype(one_type() <= two_type()) one_type_less_than_equal_two_type;
    static_assert(one_type_less_than_equal_two_type() == (1 <= 2));
    typedef decltype(one_type() >= two_type()) one_type_greater_than_equal_two_type;
    static_assert(one_type_greater_than_equal_two_type() == (1 >= 2));
    typedef decltype(one_type() == two_type()) one_type_equal_two_type;
    static_assert(one_type_equal_two_type() == (1 == 2));
    typedef decltype(one_type() != two_type()) one_type_not_equal_two_type;
    static_assert(one_type_not_equal_two_type() == (1 != 2));
    typedef decltype(one_type() & two_type()) one_type_bit_and_two_type;
    static_assert(one_type_bit_and_two_type() == (1 & 2));
    typedef decltype(one_type() ^ two_type()) one_type_xor_two_type;
    static_assert(one_type_xor_two_type() == (1 ^ 2));
    typedef decltype(one_type() | two_type()) one_type_bit_or_two_type;
    static_assert(one_type_bit_or_two_type() == (1 | 2));

};

struct test_operators_with_integrals
{
    typedef rocm::integral_constant<int, 1> one_type;
    typedef rocm::integral_constant<int, 2> two_type;

    static_assert((not rocm::true_type()) == false);

    static_assert((rocm::true_type() && rocm::false_type()) == false);
    static_assert((rocm::true_type() || rocm::false_type()) == true);

    static_assert((not two_type()) == (not 2));

// GCC confuses the complement operator with the destructor
#ifdef __clang__
    static_assert((~two_type()) == (~2));
#endif

    static_assert((+two_type()) == (+2));
    static_assert((-two_type()) == (-2));

    static_assert((one_type() + two_type()) == (1 + 2));
    static_assert((one_type() - two_type()) == (1 - 2));
    static_assert((one_type() * two_type()) == (1 * 2));
    static_assert((one_type() / two_type()) == (1 / 2));
    static_assert((one_type() % two_type()) == (1 % 2));
    static_assert((one_type() >> two_type()) == (1 >> 2));
    static_assert((one_type() << two_type()) == (1 << 2));
    static_assert((one_type() > two_type()) == (1 > 2));
    static_assert((one_type() < two_type()) == (1 < 2));
    static_assert((one_type() <= two_type()) == (1 <= 2));
    static_assert((one_type() >= two_type()) == (1 >= 2));
    static_assert((one_type() == two_type()) == (1 == 2));
    static_assert((one_type() != two_type()) == (1 != 2));
    static_assert((one_type() & two_type()) == (1 & 2));
    static_assert((one_type() ^ two_type()) == (1 ^ 2));
    static_assert((one_type() | two_type()) == (1 | 2));

};

DUAL_TEST_CASE()
{
    static_assert(rocm::bool_constant<true>{} == true);
    static_assert(rocm::bool_constant<false>{} == false);
    static_assert(rocm::bool_constant<true>{} == rocm::true_type);
    static_assert(rocm::bool_constant<false>{} == rocm::false_type);
}
