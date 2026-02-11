#include <rocm/array.hpp>
#include <migraphx/kernels/test.hpp>

// ---- Aggregate initialization ----

TEST_CASE(aggregate_init)
{
    rocm::array<int, 3> a = {1, 2, 3};
    EXPECT(a[0] == 1);
    EXPECT(a[1] == 2);
    EXPECT(a[2] == 3);
}

TEST_CASE(aggregate_init_single)
{
    rocm::array<int, 1> a = {42};
    EXPECT(a[0] == 42);
}

TEST_CASE(default_init)
{
    rocm::array<int, 3> a = {};
    EXPECT(a[0] == 0);
    EXPECT(a[1] == 0);
    EXPECT(a[2] == 0);
}

// ---- CTAD ----

TEST_CASE(ctad)
{
    rocm::array a = {1, 2, 3, 4};
    EXPECT(a.size() == 4);
    EXPECT(a[0] == 1);
    EXPECT(a[3] == 4);
}

TEST_CASE(ctad_single)
{
    rocm::array a = {99};
    EXPECT(a.size() == 1);
    EXPECT(a[0] == 99);
}

// ---- size / max_size / empty ----

TEST_CASE(size)
{
    rocm::array<int, 5> a = {1, 2, 3, 4, 5};
    EXPECT(a.size() == 5);
    EXPECT(a.max_size() == 5);
    EXPECT(not a.empty());
}

TEST_CASE(size_zero)
{
    rocm::array<int, 0> a = {};
    EXPECT(a.size() == 0);
    EXPECT(a.max_size() == 0);
    EXPECT(a.empty());
}

// ---- element access: operator[] ----

TEST_CASE(subscript_read)
{
    rocm::array<int, 4> a = {10, 20, 30, 40};
    EXPECT(a[0] == 10);
    EXPECT(a[3] == 40);
}

TEST_CASE(subscript_write)
{
    rocm::array<int, 3> a = {1, 2, 3};
    a[1] = 99;
    EXPECT(a[1] == 99);
}

TEST_CASE(subscript_const)
{
    const rocm::array<int, 3> a = {5, 6, 7};
    EXPECT(a[0] == 5);
    EXPECT(a[2] == 7);
}

// ---- element access: at ----

TEST_CASE(at_read)
{
    rocm::array<int, 3> a = {10, 20, 30};
    EXPECT(a.at(0) == 10);
    EXPECT(a.at(2) == 30);
}

TEST_CASE(at_write)
{
    rocm::array<int, 3> a = {1, 2, 3};
    a.at(0) = 77;
    EXPECT(a.at(0) == 77);
}

// ---- element access: front / back ----

TEST_CASE(front)
{
    rocm::array<int, 4> a = {10, 20, 30, 40};
    EXPECT(a.front() == 10);
}

TEST_CASE(back)
{
    rocm::array<int, 4> a = {10, 20, 30, 40};
    EXPECT(a.back() == 40);
}

TEST_CASE(front_write)
{
    rocm::array<int, 3> a = {1, 2, 3};
    a.front() = 99;
    EXPECT(a[0] == 99);
}

TEST_CASE(back_write)
{
    rocm::array<int, 3> a = {1, 2, 3};
    a.back() = 99;
    EXPECT(a[2] == 99);
}

TEST_CASE(front_back_const)
{
    const rocm::array<int, 3> a = {5, 6, 7};
    EXPECT(a.front() == 5);
    EXPECT(a.back() == 7);
}

// ---- data ----

TEST_CASE(data_non_null)
{
    rocm::array<int, 3> a = {1, 2, 3};
    EXPECT(a.data() != nullptr);
    EXPECT(a.data() == a.begin());
}

TEST_CASE(data_const)
{
    const rocm::array<int, 3> a = {1, 2, 3};
    EXPECT(a.data() != nullptr);
    EXPECT(*a.data() == 1);
}

TEST_CASE(data_zero_size)
{
    rocm::array<int, 0> a = {};
    EXPECT(a.data() == nullptr);
}

// ---- fill ----

TEST_CASE(fill)
{
    rocm::array<int, 4> a = {};
    a.fill(42);
    EXPECT(a[0] == 42);
    EXPECT(a[1] == 42);
    EXPECT(a[2] == 42);
    EXPECT(a[3] == 42);
}

TEST_CASE(fill_overwrite)
{
    rocm::array<int, 3> a = {1, 2, 3};
    a.fill(0);
    EXPECT(a[0] == 0);
    EXPECT(a[1] == 0);
    EXPECT(a[2] == 0);
}

// ---- swap ----

TEST_CASE(swap_member)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {4, 5, 6};
    a.swap(b);
    EXPECT(a[0] == 4);
    EXPECT(a[1] == 5);
    EXPECT(a[2] == 6);
    EXPECT(b[0] == 1);
    EXPECT(b[1] == 2);
    EXPECT(b[2] == 3);
}

TEST_CASE(swap_free)
{
    rocm::array<int, 3> a = {10, 20, 30};
    rocm::array<int, 3> b = {40, 50, 60};
    rocm::swap(a, b);
    EXPECT(a[0] == 40);
    EXPECT(b[0] == 10);
}

// ---- iterators: begin / end ----

TEST_CASE(begin_end)
{
    rocm::array<int, 3> a = {1, 2, 3};
    auto it               = a.begin();
    EXPECT(*it == 1);
    ++it;
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 3);
    ++it;
    EXPECT(it == a.end());
}

TEST_CASE(begin_end_const)
{
    const rocm::array<int, 3> a = {10, 20, 30};
    auto it                     = a.begin();
    EXPECT(*it == 10);
    EXPECT(*(a.end() - 1) == 30);
}

TEST_CASE(cbegin_cend)
{
    rocm::array<int, 3> a = {1, 2, 3};
    auto it                = a.cbegin();
    EXPECT(*it == 1);
    EXPECT(a.cend() - a.cbegin() == 3);
}

// ---- iterators: rbegin / rend ----

TEST_CASE(rbegin_rend)
{
    rocm::array<int, 4> a = {10, 20, 30, 40};
    auto it                = a.rbegin();
    EXPECT(*it == 40);
    ++it;
    EXPECT(*it == 30);
    ++it;
    EXPECT(*it == 20);
    ++it;
    EXPECT(*it == 10);
    ++it;
    EXPECT(it == a.rend());
}

TEST_CASE(crbegin_crend)
{
    const rocm::array<int, 3> a = {5, 6, 7};
    auto it                     = a.crbegin();
    EXPECT(*it == 7);
    EXPECT(a.crend() - a.crbegin() == 3);
}

// ---- iterators: zero-size ----

TEST_CASE(iterators_zero_size)
{
    rocm::array<int, 0> a = {};
    EXPECT(a.begin() == a.end());
    EXPECT(a.cbegin() == a.cend());
}

// ---- comparison: operator== / operator!= ----

TEST_CASE(equal)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {1, 2, 3};
    EXPECT(a == b);
    EXPECT(not(a != b));
}

TEST_CASE(not_equal)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {1, 2, 4};
    EXPECT(a != b);
    EXPECT(not(a == b));
}

TEST_CASE(equal_zero_size)
{
    rocm::array<int, 0> a = {};
    rocm::array<int, 0> b = {};
    EXPECT(a == b);
    EXPECT(not(a != b));
}

// ---- comparison: operator< / operator> / operator<= / operator>= ----

TEST_CASE(less_than)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {1, 2, 4};
    EXPECT(a < b);
    EXPECT(not(b < a));
    EXPECT(not(a < a));
}

TEST_CASE(greater_than)
{
    rocm::array<int, 3> a = {1, 2, 4};
    rocm::array<int, 3> b = {1, 2, 3};
    EXPECT(a > b);
    EXPECT(not(b > a));
}

TEST_CASE(less_equal)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {1, 2, 4};
    rocm::array<int, 3> c = {1, 2, 3};
    EXPECT(a <= b);
    EXPECT(a <= c);
    EXPECT(not(b <= a));
}

TEST_CASE(greater_equal)
{
    rocm::array<int, 3> a = {1, 2, 4};
    rocm::array<int, 3> b = {1, 2, 3};
    rocm::array<int, 3> c = {1, 2, 4};
    EXPECT(a >= b);
    EXPECT(a >= c);
    EXPECT(not(b >= a));
}

TEST_CASE(less_first_element)
{
    rocm::array<int, 3> a = {0, 9, 9};
    rocm::array<int, 3> b = {1, 0, 0};
    EXPECT(a < b);
}

// ---- to_array ----

TEST_CASE(to_array_lvalue)
{
    int c_arr[] = {10, 20, 30};
    auto a      = rocm::to_array(c_arr);
    EXPECT(a.size() == 3);
    EXPECT(a[0] == 10);
    EXPECT(a[1] == 20);
    EXPECT(a[2] == 30);
}

TEST_CASE(to_array_const)
{
    const int c_arr[] = {5, 6};
    auto a            = rocm::to_array(c_arr);
    EXPECT(a.size() == 2);
    EXPECT(a[0] == 5);
    EXPECT(a[1] == 6);
}

// ---- constexpr evaluation ----

TEST_CASE(constexpr_size)
{
    static_assert(rocm::array<int, 5>{}.size() == 5);
    static_assert(rocm::array<int, 0>{}.size() == 0);
}

TEST_CASE(constexpr_empty)
{
    static_assert(not rocm::array<int, 3>{}.empty());
    static_assert(rocm::array<int, 0>{}.empty());
}

TEST_CASE(constexpr_element_access)
{
    static constexpr rocm::array<int, 3> a = {10, 20, 30};
    static_assert(a[0] == 10);
    static_assert(a[1] == 20);
    static_assert(a[2] == 30);
    static_assert(a.front() == 10);
    static_assert(a.back() == 30);
    static_assert(a.at(1) == 20);
}

TEST_CASE(constexpr_comparison)
{
    static constexpr rocm::array<int, 3> a = {1, 2, 3};
    static constexpr rocm::array<int, 3> b = {1, 2, 3};
    static constexpr rocm::array<int, 3> c = {1, 2, 4};
    static_assert(a == b);
    static_assert(a != c);
    static_assert(a < c);
    static_assert(c > a);
    static_assert(a <= b);
    static_assert(a >= b);
}

TEST_CASE(constexpr_iterator)
{
    static constexpr rocm::array<int, 3> a = {10, 20, 30};
    static_assert(*a.begin() == 10);
    static_assert(*(a.end() - 1) == 30);
    static_assert(a.end() - a.begin() == 3);
}

TEST_CASE(constexpr_reverse_iterator)
{
    static constexpr rocm::array<int, 3> a = {10, 20, 30};
    static_assert(*a.rbegin() == 30);
    static_assert(*(a.rend() - 1) == 10);
}

// ---- type aliases ----

TEST_CASE(type_aliases)
{
    EXPECT(rocm::is_same<rocm::array<int, 3>::value_type, int>{});
    EXPECT(rocm::is_same<rocm::array<int, 3>::pointer, int*>{});
    EXPECT(rocm::is_same<rocm::array<int, 3>::const_pointer, const int*>{});
    EXPECT(rocm::is_same<rocm::array<int, 3>::reference, int&>{});
    EXPECT(rocm::is_same<rocm::array<int, 3>::const_reference, const int&>{});
    EXPECT(rocm::is_same<rocm::array<int, 3>::size_type, rocm::size_t>{});
    EXPECT(rocm::is_same<rocm::array<int, 3>::difference_type, rocm::ptrdiff_t>{});
}

// ---- iterate and sum ----

TEST_CASE(iterate_forward)
{
    rocm::array<int, 5> a = {1, 2, 3, 4, 5};
    int sum                = 0;
    for(auto it = a.begin(); it != a.end(); ++it)
        sum += *it;
    EXPECT(sum == 15);
}

TEST_CASE(iterate_reverse)
{
    rocm::array<int, 4> a = {10, 20, 30, 40};
    int result             = 0;
    int factor             = 1;
    for(auto it = a.rbegin(); it != a.rend(); ++it)
    {
        result += *it * factor;
        factor *= 10;
    }
    // 40*1 + 30*10 + 20*100 + 10*1000 = 40 + 300 + 2000 + 10000 = 12340
    EXPECT(result == 12340);
}

// ---- different element types ----

TEST_CASE(char_array)
{
    rocm::array<char, 3> a = {'a', 'b', 'c'};
    EXPECT(a[0] == 'a');
    EXPECT(a[2] == 'c');
    EXPECT(a.size() == 3);
}

TEST_CASE(long_array)
{
    rocm::array<long, 2> a = {100000L, 200000L};
    EXPECT(a.front() == 100000L);
    EXPECT(a.back() == 200000L);
}
