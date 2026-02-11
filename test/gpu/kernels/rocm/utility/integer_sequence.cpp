#include <rocm/utility/integer_sequence.hpp>
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

// ---- integer_sequence: value_type ----

TEST_CASE(integer_sequence_value_type_int)
{
    EXPECT(rocm::is_same<rocm::integer_sequence<int, 0, 1, 2>::value_type, int>{});
}

TEST_CASE(integer_sequence_value_type_long)
{
    EXPECT(rocm::is_same<rocm::integer_sequence<long, 0, 1>::value_type, long>{});
}

TEST_CASE(integer_sequence_value_type_char)
{
    EXPECT(rocm::is_same<rocm::integer_sequence<char, 'a', 'b'>::value_type, char>{});
}

TEST_CASE(integer_sequence_value_type_size_t)
{
    EXPECT(rocm::is_same<rocm::integer_sequence<rocm::size_t, 0, 1>::value_type, rocm::size_t>{});
}

// ---- integer_sequence: size ----

TEST_CASE(integer_sequence_size_0)
{
    static_assert(rocm::integer_sequence<int>::size() == 0);
}

TEST_CASE(integer_sequence_size_1)
{
    static_assert(rocm::integer_sequence<int, 42>::size() == 1);
}

TEST_CASE(integer_sequence_size_3)
{
    static_assert(rocm::integer_sequence<int, 10, 20, 30>::size() == 3);
}

TEST_CASE(integer_sequence_size_5)
{
    static_assert(rocm::integer_sequence<int, 0, 1, 2, 3, 4>::size() == 5);
}

// ---- index_sequence ----

TEST_CASE(index_sequence_is_integer_sequence)
{
    EXPECT(rocm::is_same<rocm::index_sequence<0, 1, 2>,
                         rocm::integer_sequence<rocm::size_t, 0, 1, 2>>{});
}

TEST_CASE(index_sequence_empty)
{
    EXPECT(
        rocm::is_same<rocm::index_sequence<>, rocm::integer_sequence<rocm::size_t>>{});
}

TEST_CASE(index_sequence_size)
{
    static_assert(rocm::index_sequence<0, 1, 2, 3>::size() == 4);
}

TEST_CASE(index_sequence_value_type)
{
    EXPECT(rocm::is_same<rocm::index_sequence<0>::value_type, rocm::size_t>{});
}

// ---- make_integer_sequence ----

TEST_CASE(make_integer_sequence_0)
{
    EXPECT(rocm::is_same<rocm::make_integer_sequence<int, 0>, rocm::integer_sequence<int>>{});
}

TEST_CASE(make_integer_sequence_1)
{
    EXPECT(
        rocm::is_same<rocm::make_integer_sequence<int, 1>, rocm::integer_sequence<int, 0>>{});
}

TEST_CASE(make_integer_sequence_3)
{
    EXPECT(rocm::is_same<rocm::make_integer_sequence<int, 3>,
                         rocm::integer_sequence<int, 0, 1, 2>>{});
}

TEST_CASE(make_integer_sequence_5)
{
    EXPECT(rocm::is_same<rocm::make_integer_sequence<int, 5>,
                         rocm::integer_sequence<int, 0, 1, 2, 3, 4>>{});
}

TEST_CASE(make_integer_sequence_long)
{
    EXPECT(rocm::is_same<rocm::make_integer_sequence<long, 3>,
                         rocm::integer_sequence<long, 0, 1, 2>>{});
}

TEST_CASE(make_integer_sequence_unsigned)
{
    EXPECT(rocm::is_same<rocm::make_integer_sequence<unsigned, 4>,
                         rocm::integer_sequence<unsigned, 0, 1, 2, 3>>{});
}

// ---- make_index_sequence ----

TEST_CASE(make_index_sequence_0)
{
    EXPECT(rocm::is_same<rocm::make_index_sequence<0>, rocm::index_sequence<>>{});
}

TEST_CASE(make_index_sequence_1)
{
    EXPECT(rocm::is_same<rocm::make_index_sequence<1>, rocm::index_sequence<0>>{});
}

TEST_CASE(make_index_sequence_4)
{
    EXPECT(rocm::is_same<rocm::make_index_sequence<4>, rocm::index_sequence<0, 1, 2, 3>>{});
}

TEST_CASE(make_index_sequence_size)
{
    static_assert(rocm::make_index_sequence<6>::size() == 6);
}

// ---- index_sequence_for ----

TEST_CASE(index_sequence_for_empty)
{
    EXPECT(rocm::is_same<rocm::index_sequence_for<>, rocm::index_sequence<>>{});
}

TEST_CASE(index_sequence_for_one)
{
    EXPECT(rocm::is_same<rocm::index_sequence_for<int>, rocm::index_sequence<0>>{});
}

TEST_CASE(index_sequence_for_three)
{
    EXPECT(rocm::is_same<rocm::index_sequence_for<int, float, double>,
                         rocm::index_sequence<0, 1, 2>>{});
}

TEST_CASE(index_sequence_for_size)
{
    static_assert(rocm::index_sequence_for<int, int, int, int>::size() == 4);
}
