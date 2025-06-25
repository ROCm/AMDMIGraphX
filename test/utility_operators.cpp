
#include <migraphx/utility_operators.hpp>
#include <test.hpp>

// template <class Expression>
// static auto compare_predicate(const Expression& e)
// {
//     bool result = e.value();
//     return test::make_predicate(test::as_string(e) + " => " + test::as_string(result),
//                                 [=] { return result; });
// }

// NOLINTNEXTLINE
#define MIGRAPHX_OP_TEST_COMPARE(...) (__VA_ARGS__)
// #define MIGRAPHX_OP_TEST_COMPARE(...) compare_predicate(TEST_CAPTURE(__VA_ARGS__))

// NOLINTNEXTLINE
#define EXPECT_TOTALLY_ORDERED_IMPL(_, x, y)     \
    EXPECT(_(x <= y) or _(x >= y));              \
    EXPECT(_(x < y) or _(x > y) or _(x == y));   \
    EXPECT((_(x < y) or _(x > y)) == _(x != y)); \
    EXPECT(_(x < y) == _(y > x));                \
    EXPECT(_(x <= y) == _(y >= x));              \
    EXPECT(_(x < y) != _(x >= y));               \
    EXPECT(_(x > y) != _(x <= y));               \
    EXPECT(_(x == y) != _(x != y))

// NOLINTNEXTLINE
#define EXPECT_TOTALLY_ORDERED(x, y)                                \
    EXPECT_TOTALLY_ORDERED_IMPL(MIGRAPHX_OP_TEST_COMPARE, x, y); \
    EXPECT_TOTALLY_ORDERED_IMPL(MIGRAPHX_OP_TEST_COMPARE, y, x)


struct custom_compare_any : migraphx::totally_ordered<custom_compare_any>
{
    int x;

    constexpr custom_compare_any(int px) : x(px) {}

    constexpr bool operator==(const custom_compare_any& rhs) const
    {
        return x == rhs.x;
    }

    template<class T>
    constexpr auto operator==(const T& rhs) const -> decltype(std::declval<int>() == rhs)
    {
        return x == rhs;
    }

    constexpr bool operator<(const custom_compare_any& rhs) const
    {
        return x < rhs.x;
    }
    
    template<class T>
    constexpr auto operator<(const T& rhs) const -> decltype(std::declval<int>() < rhs)
    {
        return x < rhs;
    }

    template<class T>
    constexpr auto operator>(const T& rhs) const -> decltype(std::declval<int>() > rhs)
    {
        return x > rhs;
    }

    template<class Stream>
    friend Stream& operator<<(Stream& os, const custom_compare_any& self)
    {
        return os << self.x;
    }
};

TEST_CASE(compare_any)
{
    custom_compare_any x{1};
    custom_compare_any y{2};
    EXPECT(1 == x);
    EXPECT(x != y);
    EXPECT(y > x);
    EXPECT(x == x);
    EXPECT(x < 2);
    EXPECT(y > 1);

    EXPECT_TOTALLY_ORDERED(x, x);
    EXPECT_TOTALLY_ORDERED(x, y);
    EXPECT_TOTALLY_ORDERED(x, 1);
}

// template<class T>
// struct custom_compare_template : migraphx::totally_ordered<custom_compare_template<T>>
// {
//     T x;

//     constexpr custom_compare_template(T px) : x(px) {}

//     constexpr bool operator==(const custom_compare_template& rhs) const
//     {
//         return x == rhs.x;
//     }

//     template<class U>
//     constexpr auto operator==(const custom_compare_template<U>& rhs) const
//     {
//         return x == rhs.x;
//     }

//     constexpr bool operator<(const custom_compare_template& rhs) const
//     {
//         return x < rhs.x;
//     }
    
//     template<class U>
//     constexpr auto operator<(const custom_compare_template<U>& rhs) const
//     {
//         return x < rhs.x;
//     }

//     template<class U>
//     constexpr auto operator>(const custom_compare_template<U>& rhs) const
//     {
//         return x > rhs.x;
//     }

//     template<class Stream>
//     friend Stream& operator<<(Stream& os, const custom_compare_template& self)
//     {
//         return os << self.x;
//     }
// };

// TEST_CASE(compare_template)
// {
//     custom_compare_template<int> x{1};
//     custom_compare_template<int> y{2};
//     custom_compare_template<short> z{2};
//     EXPECT(x != y);
//     EXPECT(x == x);
//     EXPECT(y > x);
//     EXPECT(x > y);
//     EXPECT(z == z);
//     EXPECT(z == x);
//     EXPECT(z != y);
//     EXPECT(z > x);
//     EXPECT(x > z);

//     EXPECT_TOTALLY_ORDERED(x, x);
//     EXPECT_TOTALLY_ORDERED(x, y);
//     EXPECT_TOTALLY_ORDERED(x, z);
//     EXPECT_TOTALLY_ORDERED(y, z);
// }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
