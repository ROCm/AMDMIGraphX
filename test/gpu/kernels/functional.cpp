/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/test.hpp>

template <class T>
constexpr T test_lift_add(T x, T y)
{
    return x + y;
}

MIGRAPHX_LIFT_CLASS(test_lift_add_class, test_lift_add);

template <class T>
constexpr auto test_returns_twice(T x) MIGRAPHX_RETURNS(x + x);

TEST_CASE(swallow_ignores_arguments)
{
    [[maybe_unused]] migraphx::swallow s{1, 2.0f, 'c'};
    [[maybe_unused]] migraphx::ignore<3> i{42};
    EXPECT(true);
}

TEST_CASE(overload_dispatch)
{
    auto f = migraphx::overload([](int) { return 1; }, [](float) { return 2; });
    EXPECT(f(3) == 1);
    EXPECT(f(3.0f) == 2);
}

TEST_CASE(always_returns_value)
{
    auto f = migraphx::always(42);
    EXPECT(f() == 42);
    EXPECT(f(1, 2.0, 'x') == 42);
}

TEST_CASE(sequence_c_indices)
{
    auto sum = migraphx::sequence_c<4>([](auto... is) { return (0 + ... + is); });
    EXPECT(sum == 6);
}

TEST_CASE(sequence_c_count)
{
    auto n = migraphx::sequence_c<5>([](auto... is) { return sizeof...(is); });
    EXPECT(n == 5u);
}

TEST_CASE(sequence_c_zero)
{
    auto n = migraphx::sequence_c<0>([](auto... is) { return sizeof...(is); });
    EXPECT(n == 0u);
}

TEST_CASE(sequence_c_order)
{
    auto v = migraphx::sequence_c<3>([](auto... is) {
        return migraphx::fold([](auto x, auto y) { return x * 10 + y; })(0, is...);
    });
    EXPECT(v == 12);
}

TEST_CASE(sequence_with_constant)
{
    auto n = migraphx::sequence(migraphx::_c<3>, [](auto... is) { return sizeof...(is); });
    EXPECT(n == 3u);
}

TEST_CASE(by_transform_then_combine)
{
    auto f = migraphx::by([](auto x) { return x * 2; }, [](auto... xs) { return (0 + ... + xs); });
    EXPECT(f(1, 2, 3) == 12);
}

TEST_CASE(by_void_combine)
{
    int total = 0;
    auto f    = migraphx::by([](auto x) { return x; }, [&](auto... xs) { total = (0 + ... + xs); });
    f(1, 2, 3);
    EXPECT(total == 6);
}

TEST_CASE(by_single_function)
{
    int sum = 0;
    auto f  = migraphx::by([&](auto x) { sum += x; });
    f(1, 2, 3);
    EXPECT(sum == 6);
}

TEST_CASE(each_args_sum)
{
    int sum = 0;
    migraphx::each_args([&](auto x) { sum += x; }, 1, 2, 3);
    EXPECT(sum == 6);
}

TEST_CASE(each_args_order)
{
    int v = 0;
    migraphx::each_args([&](auto x) { v = v * 10 + x; }, 1, 2, 3);
    EXPECT(v == 123);
}

TEST_CASE(each_args_empty)
{
    int count = 0;
    migraphx::each_args([&](auto) { count++; });
    EXPECT(count == 0);
}

TEST_CASE(unpack_each_single_pack)
{
    int sum = 0;
    migraphx::unpack_each([&](auto x) { sum += x; }, migraphx::pack(1, 2, 3));
    EXPECT(sum == 6);
}

TEST_CASE(unpack_each_two_packs)
{
    int sum = 0;
    migraphx::unpack_each(
        [&](auto x, auto y) { sum += x * y; }, migraphx::pack(1, 2), migraphx::pack(3, 4));
    EXPECT(sum == 11);
}

TEST_CASE(unpack_each_three_packs)
{
    int sum = 0;
    migraphx::unpack_each([&](auto x, auto y, auto z) { sum += x * 100 + y * 10 + z; },
                          migraphx::pack(1, 2),
                          migraphx::pack(3, 4),
                          migraphx::pack(5, 6));
    EXPECT(sum == 762);
}

TEST_CASE(repeat_c_count)
{
    int count = 0;
    migraphx::repeat_c<5>([&](auto) { count++; });
    EXPECT(count == 5);
}

TEST_CASE(repeat_c_indices)
{
    int v = 0;
    migraphx::repeat_c<3>([&](auto i) { v = v * 10 + i; });
    EXPECT(v == 12);
}

TEST_CASE(repeat_with_constant)
{
    int count = 0;
    migraphx::repeat(migraphx::_c<4>, [&](auto) { count++; });
    EXPECT(count == 4);
}

TEST_CASE(repeat_up_by_2)
{
    int v = 0;
    migraphx::repeat_up_by_2_c<1, 16>([&](auto i) { v = v * 100 + i; });
    EXPECT(v == 1020408);
}

TEST_CASE(repeat_up_by_2_default_start)
{
    int sum = 0;
    migraphx::repeat_up_by_2_c<8>([&](auto i) { sum += i; });
    EXPECT(sum == 7);
}

TEST_CASE(repeat_up_by_2_non_power)
{
    int sum = 0;
    migraphx::repeat_up_by_2_c<1, 5>([&](auto i) { sum += i; });
    EXPECT(sum == 7);
}

TEST_CASE(repeat_down_by_2)
{
    int v = 0;
    migraphx::repeat_down_by_2_c<8, 1>([&](auto i) { v = v * 100 + i; });
    EXPECT(v == 8040201);
}

TEST_CASE(repeat_down_by_2_default_last)
{
    int sum = 0;
    migraphx::repeat_down_by_2_c<4>([&](auto i) { sum += i; });
    EXPECT(sum == 7);
}

TEST_CASE(repeat_down_by_2_partial)
{
    int sum = 0;
    migraphx::repeat_down_by_2_c<8, 2>([&](auto i) { sum += i; });
    EXPECT(sum == 14);
}

TEST_CASE(fold_sum)
{
    EXPECT(migraphx::fold([](auto x, auto y) { return x + y; })(1, 2, 3, 4) == 10);
}

TEST_CASE(fold_left_associative)
{
    EXPECT(migraphx::fold([](auto x, auto y) { return x - y; })(10, 1, 2) == 7);
}

TEST_CASE(fold_single)
{
    EXPECT(migraphx::fold([](auto x, auto y) { return x + y; })(5) == 5);
}

TEST_CASE(compose_two)
{
    auto f = migraphx::compose([](int x) { return x + 1; }, [](int x) { return x * 2; });
    EXPECT(f(3) == 7);
}

TEST_CASE(compose_three)
{
    auto f = migraphx::compose(
        [](int x) { return x + 1; }, [](int x) { return x * 2; }, [](int x) { return x + 3; });
    EXPECT(f(1) == 9);
}

TEST_CASE(compose_single)
{
    auto f = migraphx::compose([](int x) { return x * 3; });
    EXPECT(f(4) == 12);
}

TEST_CASE(partial_apply)
{
    auto add3 = [](int a, int b, int c) { return a * 100 + b * 10 + c; };
    EXPECT(migraphx::partial(add3)(1, 2)(3) == 123);
    EXPECT(migraphx::partial(add3)(1)(2, 3) == 123);
    EXPECT(migraphx::partial(add3)()(1, 2, 3) == 123);
}

TEST_CASE(pack_apply)
{
    auto p = migraphx::pack(1, 2, 3);
    EXPECT(p([](auto... xs) { return (0 + ... + xs); }) == 6);
}

TEST_CASE(pack_empty)
{
    auto p = migraphx::pack();
    EXPECT(p([](auto... xs) { return sizeof...(xs); }) == 0u);
}

TEST_CASE(pack_forward_references)
{
    int x = 1;
    int y = 2;
    migraphx::pack_forward(x, y)([](auto&& a, auto&& b) {
        a = 10;
        b = 20;
    });
    EXPECT(x == 10);
    EXPECT(y == 20);
}

TEST_CASE(join_two_packs)
{
    auto r = migraphx::join(
        [](auto... xs) { return (0 + ... + xs); }, migraphx::pack(1, 2), migraphx::pack(3, 4));
    EXPECT(r == 10);
}

TEST_CASE(join_single_pack)
{
    auto r = migraphx::join([](auto... xs) { return (0 + ... + xs); }, migraphx::pack(1, 2, 3));
    EXPECT(r == 6);
}

TEST_CASE(join_order)
{
    auto r = migraphx::join(
        [](auto... xs) {
            return migraphx::fold([](auto x, auto y) { return x * 10 + y; })(0, xs...);
        },
        migraphx::pack(1),
        migraphx::pack(2),
        migraphx::pack(3));
    EXPECT(r == 123);
}

TEST_CASE(pack_compare_lexicographical)
{
    auto lt = [](auto x, auto y) { return x < y; };
    EXPECT(migraphx::pack_compare(lt, migraphx::pack(1, 2), migraphx::pack(1, 3)) == 1);
    EXPECT(migraphx::pack_compare(lt, migraphx::pack(1, 3), migraphx::pack(1, 2)) == -1);
    EXPECT(migraphx::pack_compare(lt, migraphx::pack(1, 2), migraphx::pack(1, 2)) == 0);
    EXPECT(migraphx::pack_compare(lt, migraphx::pack(2, 0), migraphx::pack(1, 9)) == -1);
}

TEST_CASE(arg_c_select)
{
    EXPECT(migraphx::arg_c<0>()(1, 2, 3) == 1);
    EXPECT(migraphx::arg_c<1>()(1, 2, 3) == 2);
    EXPECT(migraphx::arg_c<2>()(1, 2, 3) == 3);
}

TEST_CASE(arg_select) { EXPECT(migraphx::arg(migraphx::_c<1>)(4, 5, 6) == 5); }

TEST_CASE(make_transform_basic)
{
    auto t = migraphx::make_transform([](auto f, auto... xs) { return f(xs..., 10); });
    auto r = t(1, 2)([](auto... xs) { return (0 + ... + xs); });
    EXPECT(r == 13);
}

TEST_CASE(transform_args_identity)
{
    auto r = migraphx::transform_args()(1, 2, 3)([](auto... xs) { return (0 + ... + xs); });
    EXPECT(r == 6);
}

TEST_CASE(transform_args_single)
{
    auto append = migraphx::make_transform([](auto f, auto... xs) { return f(xs..., 7); });
    auto r      = migraphx::transform_args(append)(1, 2)([](auto... xs) { return (0 + ... + xs); });
    EXPECT(r == 10);
}

TEST_CASE(transform_args_chain)
{
    auto append7     = migraphx::make_transform([](auto f, auto... xs) { return f(xs..., 7); });
    auto double_each = migraphx::make_transform([](auto f, auto... xs) { return f((xs * 2)...); });
    auto r = migraphx::transform_args(append7,
                                      double_each)(1, 2)([](auto... xs) { return (0 + ... + xs); });
    EXPECT(r == 20);
}

TEST_CASE(transform_args_chain_order)
{
    auto append7     = migraphx::make_transform([](auto f, auto... xs) { return f(xs..., 7); });
    auto double_each = migraphx::make_transform([](auto f, auto... xs) { return f((xs * 2)...); });
    auto r = migraphx::transform_args(double_each,
                                      append7)(1, 2)([](auto... xs) { return (0 + ... + xs); });
    EXPECT(r == 13);
}

TEST_CASE(rotate_last_one)
{
    auto r = migraphx::rotate_last()(1, 2, 3)(
        [](auto x, auto y, auto z) { return x * 100 + y * 10 + z; });
    EXPECT(r == 312);
}

TEST_CASE(rotate_last_two)
{
    auto r = migraphx::rotate_last<2>()(1, 2, 3, 4)(
        [](auto a, auto b, auto c, auto d) { return a * 1000 + b * 100 + c * 10 + d; });
    EXPECT(r == 3412);
}

TEST_CASE(rotate_last_full_rotation)
{
    auto r = migraphx::rotate_last<3>()(1, 2, 3)(
        [](auto x, auto y, auto z) { return x * 100 + y * 10 + z; });
    EXPECT(r == 123);
}

TEST_CASE(pack_first_basic)
{
    auto r = migraphx::pack_first<2>()(1, 2, 3, 4)([](auto p, auto z, auto w) {
        return p([](auto x, auto y) { return x * 10 + y; }) * 100 + z * 10 + w;
    });
    EXPECT(r == 1234);
}

TEST_CASE(pack_first_none)
{
    auto r = migraphx::pack_first<0>()(1, 2)([](auto p, auto x, auto y) {
        return p([](auto... xs) { return sizeof...(xs); }) * 100 + x * 10 + y;
    });
    EXPECT(r == 12u);
}

TEST_CASE(rotate_and_pack_last_basic)
{
    auto r = migraphx::rotate_and_pack_last<1>()(1, 2, 3)(
        [](auto p, auto x, auto y) { return p([](auto z) { return z; }) * 100 + x * 10 + y; });
    EXPECT(r == 312);
}

TEST_CASE(returns_macro) { EXPECT(test_returns_twice(21) == 42); }

TEST_CASE(lift_function)
{
    auto f = MIGRAPHX_LIFT(test_lift_add);
    EXPECT(f(2, 3) == 5);
}

TEST_CASE(lift_pass_to_higher_order)
{
    auto r = migraphx::fold(MIGRAPHX_LIFT(test_lift_add))(1, 2, 3);
    EXPECT(r == 6);
}

TEST_CASE(lift_class_function_object)
{
    test_lift_add_class f{};
    EXPECT(f(2, 3) == 5);
    EXPECT(migraphx::fold(test_lift_add_class{})(1, 2, 3, 4) == 10);
}
