
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef MIGRAPH_GUARD_TEST_TEST_HPP
#define MIGRAPH_GUARD_TEST_TEST_HPP

namespace test {
// NOLINTNEXTLINE
#define TEST_FOREACH_OPERATOR(m)                                                                   \
    m(==, equal) m(!=, not_equal) m(<=, less_than_equal) m(>=, greater_than_equal) m(<, less_than) \
        m(>, greater_than)

// NOLINTNEXTLINE
#define TEST_EACH_OPERATOR_OBJECT(op, name)            \
    struct name                                        \
    {                                                  \
        static std::string as_string() { return #op; } \
        template <class T, class U>                    \
        static decltype(auto) call(T&& x, U&& y)       \
        {                                              \
            return x op y;                             \
        }                                              \
    };

TEST_FOREACH_OPERATOR(TEST_EACH_OPERATOR_OBJECT)

inline std::ostream& operator<<(std::ostream& s, std::nullptr_t)
{
    s << "nullptr";
    return s;
}

template <class T, class U, class Operator>
struct expression
{
    T lhs;
    U rhs;

    friend std::ostream& operator<<(std::ostream& s, const expression& self)
    {
        s << " [ " << self.lhs << " " << Operator::as_string() << " " << self.rhs << " ]";
        return s;
    }

    decltype(auto) value() const { return Operator::call(lhs, rhs); };
};

// TODO: Remove rvalue references
template <class T, class U, class Operator>
expression<T, U, Operator> make_expression(T&& rhs, U&& lhs, Operator)
{
    return {std::forward<T>(rhs), std::forward<U>(lhs)};
}

template <class T>
struct lhs_expression;

// TODO: Remove rvalue reference
template <class T>
lhs_expression<T> make_lhs_expression(T&& lhs)
{
    return lhs_expression<T>{std::forward<T>(lhs)};
}

template <class T>
struct lhs_expression
{
    T lhs;
    explicit lhs_expression(T e) : lhs(e) {}

    friend std::ostream& operator<<(std::ostream& s, const lhs_expression& self)
    {
        s << self.lhs;
        return s;
    }

    T value() const { return lhs; }
// NOLINTNEXTLINE
#define TEST_LHS_OPERATOR(op, name)                            \
    template <class U>                                         \
    auto operator op(const U& rhs) const                       \
    {                                                          \
        return make_expression(lhs, rhs, name{}); /* NOLINT */ \
    }

    TEST_FOREACH_OPERATOR(TEST_LHS_OPERATOR)
// NOLINTNEXTLINE
#define TEST_LHS_REOPERATOR(op)                 \
    template <class U>                          \
    auto operator op(const U& rhs) const        \
    {                                           \
        return make_lhs_expression(lhs op rhs); \
    }
    TEST_LHS_REOPERATOR(+)
    TEST_LHS_REOPERATOR(-)
    TEST_LHS_REOPERATOR(*)
    TEST_LHS_REOPERATOR(/)
    TEST_LHS_REOPERATOR(%)
    TEST_LHS_REOPERATOR(&)
    TEST_LHS_REOPERATOR(|)
    TEST_LHS_REOPERATOR(&&)
    TEST_LHS_REOPERATOR(||)
};

struct capture
{
    template <class T>
    auto operator->*(const T& x)
    {
        return make_lhs_expression(x);
    }
};

template <class T, class F>
void failed(T x, const char* msg, const char* func, const char* file, int line, F f)
{
    if(!x.value())
    {
        std::cout << func << std::endl;
        std::cout << file << ":" << line << ":" << std::endl;
        std::cout << "    FAILED: " << msg << " " << x << std::endl;
        f();
    }
}

template <class F>
bool throws(F f)
{
    try
    {
        f();
        return false;
    }
    catch(...)
    {
        return true;
    }
}

template <class Exception, class F>
bool throws(F f, const std::string& msg = "")
{
    try
    {
        f();
        return false;
    }
    catch(const Exception& ex)
    {
        return std::string(ex.what()).find(msg) != std::string::npos;
    }
}

template <class T>
void run_test()
{
    T t = {};
    t.run();
}

} // namespace test

// NOLINTNEXTLINE
#define CHECK(...)                                                                                 \
    test::failed(                                                                                  \
        test::capture{}->*__VA_ARGS__, #__VA_ARGS__, __PRETTY_FUNCTION__, __FILE__, __LINE__, [] { \
        })
// NOLINTNEXTLINE
#define EXPECT(...)                             \
    test::failed(test::capture{}->*__VA_ARGS__, \
                 #__VA_ARGS__,                  \
                 __PRETTY_FUNCTION__,           \
                 __FILE__,                      \
                 __LINE__,                      \
                 &std::abort)
// NOLINTNEXTLINE
#define STATUS(...) EXPECT((__VA_ARGS__) == 0)

#endif
