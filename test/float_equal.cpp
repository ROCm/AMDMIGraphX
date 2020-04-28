#include <migraphx/float_equal.hpp>
#include <migraphx/half.hpp>
#include "test.hpp"

#include <limits>

template <class T, class U>
struct float_equal_expression
{
    T lhs;
    U rhs;

    operator bool() const { return migraphx::float_equal(lhs, rhs); }

    bool operator not() const { return not bool(*this); }

    friend std::ostream& operator<<(std::ostream& s, const float_equal_expression& self)
    {
        s << "migraphx::float_equal(" << self.lhs << ", " << self.rhs << ")";
        return s;
    }
};

template <class T, class U>
auto test_float_equal(T x, U y)
{
    return test::make_lhs_expression(float_equal_expression<T, U>{x, y});
}

template <class T, class U>
void test_equality()
{
    auto x1 = T(0.1);
    auto x2 = U(0.0);
    auto x3 = U(1.0);
    EXPECT(test_float_equal(x1, x1));
    EXPECT(test_float_equal(x2, x2));
    EXPECT(test_float_equal(x3, x3));

    EXPECT(not test_float_equal(x1, x2));
    EXPECT(not test_float_equal(x2, x1));
    EXPECT(not test_float_equal(x1, x3));
    EXPECT(not test_float_equal(x3, x1));
    EXPECT(not test_float_equal(x2, x3));
    EXPECT(not test_float_equal(x3, x2));
}

TEST_CASE_REGISTER(test_equality<double, float>);
TEST_CASE_REGISTER(test_equality<double, int>);
TEST_CASE_REGISTER(test_equality<double, migraphx::half>);
TEST_CASE_REGISTER(test_equality<float, int>);
TEST_CASE_REGISTER(test_equality<migraphx::half, int>);

template <class T, class U>
void test_limits()
{
    auto max1 = std::numeric_limits<T>::max();
    auto max2 = std::numeric_limits<U>::max();

    auto min1 = std::numeric_limits<T>::lowest();
    auto min2 = std::numeric_limits<U>::lowest();

    EXPECT(test_float_equal(max1, max1));
    EXPECT(test_float_equal(max2, max2));

    EXPECT(not test_float_equal(max1, max2));
    EXPECT(not test_float_equal(max2, max1));

    EXPECT(test_float_equal(min1, min1));
    EXPECT(test_float_equal(min2, min2));

    EXPECT(not test_float_equal(min1, min2));
    EXPECT(not test_float_equal(min2, min1));

    EXPECT(not test_float_equal(max1, min1));
    EXPECT(not test_float_equal(min1, max1));
    EXPECT(not test_float_equal(max2, min2));
    EXPECT(not test_float_equal(min2, max2));

    EXPECT(not test_float_equal(max1, min2));
    EXPECT(not test_float_equal(min2, max1));

    EXPECT(not test_float_equal(max2, min1));
    EXPECT(not test_float_equal(min1, max2));
}

TEST_CASE_REGISTER(test_limits<double, float>);
TEST_CASE_REGISTER(test_limits<double, int>);
TEST_CASE_REGISTER(test_limits<double, migraphx::half>);
TEST_CASE_REGISTER(test_limits<float, int>);
TEST_CASE_REGISTER(test_limits<int, migraphx::half>);
TEST_CASE_REGISTER(test_limits<long, int>);
TEST_CASE_REGISTER(test_limits<long, char>);

int main(int argc, const char* argv[]) { test::run(argc, argv); }
