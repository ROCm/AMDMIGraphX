/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#include <type_traits_test.hpp>

struct test_operators_with_value
{
    static_assert(rocm::true_type::value);
    static_assert(not rocm::false_type::value);

    using one_type = rocm::integral_constant<int, 1>;
    using two_type = rocm::integral_constant<int, 2>;

    using not_true_type = decltype(!rocm::true_type());
    static_assert(!not_true_type::value);

    using true_type_and_false_type = decltype(rocm::true_type() && rocm::false_type());
    static_assert(!true_type_and_false_type::value);
    using true_type_or_false_type = decltype(rocm::true_type() || rocm::false_type());
    static_assert(true_type_or_false_type::value);

    using not_two_type = decltype(!two_type());
    static_assert(not_two_type::value == (not 2));

// GCC confuses the complement operator with the destructor
#ifdef __clang__
    using compl_two_type = decltype(~two_type());
    static_assert(compl_two_type::value == (~2));
#endif

    using unary_plus_two_type = decltype(+two_type());
    static_assert(unary_plus_two_type::value == (+2));
    using unary_subtract_two_type = decltype(-two_type());
    static_assert(unary_subtract_two_type::value == (-2));

    using one_type_add_two_type = decltype(one_type() + two_type());
    static_assert(one_type_add_two_type::value == (1 + 2));
    using one_type_subtract_two_type = decltype(one_type() - two_type());
    static_assert(one_type_subtract_two_type::value == (1 - 2));
    using one_type_multiply_two_type = decltype(one_type() * two_type());
    static_assert(one_type_multiply_two_type::value == (1 * 2));
    using one_type_divide_two_type = decltype(one_type() / two_type());
    static_assert(one_type_divide_two_type::value == (1 / 2));
    using one_type_remainder_two_type = decltype(one_type() % two_type());
    static_assert(one_type_remainder_two_type::value == (1 % 2));
    using one_type_shift_right_two_type = decltype(one_type() >> two_type());
    static_assert(one_type_shift_right_two_type::value == (1 >> 2));
    using one_type_shift_left_two_type = decltype(one_type() << two_type());
    static_assert(one_type_shift_left_two_type::value == (1 << 2));
    using one_type_greater_than_two_type = decltype(one_type() > two_type());
    static_assert(one_type_greater_than_two_type::value == (1 > 2));
    using one_type_less_than_two_type = decltype(one_type() < two_type());
    static_assert(one_type_less_than_two_type::value == (1 < 2));
    using one_type_less_than_equal_two_type = decltype(one_type() <= two_type());
    static_assert(one_type_less_than_equal_two_type::value == (1 <= 2));
    using one_type_greater_than_equal_two_type = decltype(one_type() >= two_type());
    static_assert(one_type_greater_than_equal_two_type::value == (1 >= 2));
    using one_type_equal_two_type = decltype(one_type() == two_type());
    static_assert(one_type_equal_two_type::value == (1 == 2));
    using one_type_not_equal_two_type = decltype(one_type() != two_type());
    static_assert(one_type_not_equal_two_type::value == (1 != 2));
    using one_type_bit_and_two_type = decltype(one_type() & two_type());
    static_assert(one_type_bit_and_two_type::value == (1 & 2));
    using one_type_xor_two_type = decltype(one_type() ^ two_type());
    static_assert(one_type_xor_two_type::value == (1 ^ 2));
    using one_type_bit_or_two_type = decltype(one_type() | two_type());
    static_assert(one_type_bit_or_two_type::value == (1 | 2));
};

struct test_operators_implicit_conversion
{
    static_assert(rocm::true_type());
    static_assert(not rocm::false_type());

    using one_type = rocm::integral_constant<int, 1>;
    using two_type = rocm::integral_constant<int, 2>;

    using not_true_type = decltype(!rocm::true_type());
    static_assert(!not_true_type());

    using true_type_and_false_type = decltype(rocm::true_type() && rocm::false_type());
    static_assert(!true_type_and_false_type());
    using true_type_or_false_type = decltype(rocm::true_type() || rocm::false_type());
    static_assert(true_type_or_false_type());

    using not_two_type = decltype(!two_type());
    static_assert(not_two_type() == (not 2));

// GCC confuses the complement operator with the destructor
#ifdef __clang__
    using compl_two_type = decltype(~two_type());
    static_assert(compl_two_type() == (~2));
#endif

    using unary_plus_two_type = decltype(+two_type());
    static_assert(unary_plus_two_type() == (+2));
    using unary_subtract_two_type = decltype(-two_type());
    static_assert(unary_subtract_two_type() == (-2));

    using one_type_add_two_type = decltype(one_type() + two_type());
    static_assert(one_type_add_two_type() == (1 + 2));
    using one_type_subtract_two_type = decltype(one_type() - two_type());
    static_assert(one_type_subtract_two_type() == (1 - 2));
    using one_type_multiply_two_type = decltype(one_type() * two_type());
    static_assert(one_type_multiply_two_type() == (1 * 2));
    using one_type_divide_two_type = decltype(one_type() / two_type());
    static_assert(one_type_divide_two_type() == (1 / 2));
    using one_type_remainder_two_type = decltype(one_type() % two_type());
    static_assert(one_type_remainder_two_type() == (1 % 2));
    using one_type_shift_right_two_type = decltype(one_type() >> two_type());
    static_assert(one_type_shift_right_two_type() == (1 >> 2));
    using one_type_shift_left_two_type = decltype(one_type() << two_type());
    static_assert(one_type_shift_left_two_type() == (1 << 2));
    using one_type_greater_than_two_type = decltype(one_type() > two_type());
    static_assert(one_type_greater_than_two_type() == (1 > 2));
    using one_type_less_than_two_type = decltype(one_type() < two_type());
    static_assert(one_type_less_than_two_type() == (1 < 2));
    using one_type_less_than_equal_two_type = decltype(one_type() <= two_type());
    static_assert(one_type_less_than_equal_two_type() == (1 <= 2));
    using one_type_greater_than_equal_two_type = decltype(one_type() >= two_type());
    static_assert(one_type_greater_than_equal_two_type() == (1 >= 2));
    using one_type_equal_two_type = decltype(one_type() == two_type());
    static_assert(one_type_equal_two_type() == (1 == 2));
    using one_type_not_equal_two_type = decltype(one_type() != two_type());
    static_assert(one_type_not_equal_two_type() == (1 != 2));
    using one_type_bit_and_two_type = decltype(one_type() & two_type());
    static_assert(one_type_bit_and_two_type() == (1 & 2));
    using one_type_xor_two_type = decltype(one_type() ^ two_type());
    static_assert(one_type_xor_two_type() == (1 ^ 2));
    using one_type_bit_or_two_type = decltype(one_type() | two_type());
    static_assert(one_type_bit_or_two_type() == (1 | 2));
};

struct test_operators_with_integrals
{
    using one_type = rocm::integral_constant<int, 1>;
    using two_type = rocm::integral_constant<int, 2>;

    static_assert(!(not rocm::true_type()));

    static_assert(!(rocm::true_type() && rocm::false_type()));
    static_assert((rocm::true_type() || rocm::false_type()));

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

ROCM_DUAL_TEST_CASE()
{
    static_assert(rocm::bool_constant<true>{});
    static_assert(!rocm::bool_constant<false>{});
    static_assert(rocm::bool_constant<true>{} == rocm::true_type{});
    static_assert(rocm::bool_constant<false>{} == rocm::false_type{});
}
