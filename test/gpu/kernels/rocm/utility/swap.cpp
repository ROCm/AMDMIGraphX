#include <rocm/utility/swap.hpp>
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

// ---- basic swap ----

TEST_CASE(swap_int)
{
    int a = 1;
    int b = 2;
    rocm::swap(a, b);
    EXPECT(a == 2);
    EXPECT(b == 1);
}

TEST_CASE(swap_same_value)
{
    int a = 42;
    int b = 42;
    rocm::swap(a, b);
    EXPECT(a == 42);
    EXPECT(b == 42);
}

TEST_CASE(swap_negative)
{
    int a = -10;
    int b = 20;
    rocm::swap(a, b);
    EXPECT(a == 20);
    EXPECT(b == -10);
}

TEST_CASE(swap_zero)
{
    int a = 0;
    int b = 99;
    rocm::swap(a, b);
    EXPECT(a == 99);
    EXPECT(b == 0);
}

// ---- different types ----

TEST_CASE(swap_long)
{
    long a = 100000L;
    long b = 200000L;
    rocm::swap(a, b);
    EXPECT(a == 200000L);
    EXPECT(b == 100000L);
}

TEST_CASE(swap_char)
{
    char a = 'x';
    char b = 'y';
    rocm::swap(a, b);
    EXPECT(a == 'y');
    EXPECT(b == 'x');
}

TEST_CASE(swap_bool)
{
    bool a = true;
    bool b = false;
    rocm::swap(a, b);
    EXPECT(not a);
    EXPECT(b);
}

// ---- pointer swap ----

TEST_CASE(swap_pointer)
{
    int x  = 10;
    int y  = 20;
    int* a = &x;
    int* b = &y;
    rocm::swap(a, b);
    EXPECT(*a == 20);
    EXPECT(*b == 10);
}

// ---- constexpr swap ----

TEST_CASE(constexpr_swap)
{
    // swap is constexpr, verify via a constexpr lambda
    static_assert([] {
        int a = 5;
        int b = 10;
        rocm::swap(a, b);
        return a == 10 and b == 5;
    }());
}

// ---- swap preserves other elements ----

TEST_CASE(swap_array_elements)
{
    int arr[] = {1, 2, 3, 4, 5};
    rocm::swap(arr[0], arr[4]);
    EXPECT(arr[0] == 5);
    EXPECT(arr[4] == 1);
    EXPECT(arr[1] == 2);
    EXPECT(arr[2] == 3);
    EXPECT(arr[3] == 4);
}
