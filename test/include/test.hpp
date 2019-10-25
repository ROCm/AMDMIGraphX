
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

#ifndef MIGRAPHX_GUARD_TEST_TEST_HPP
#define MIGRAPHX_GUARD_TEST_TEST_HPP

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

template <class T>
inline std::ostream& operator<<(std::ostream& s, const std::vector<T>& v)
{
    s << "{ ";
    for(auto&& x : v)
    {
        s << x << ", ";
    }
    s << "}";
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
    TEST_LHS_REOPERATOR(^)
    TEST_LHS_REOPERATOR(&&)
    TEST_LHS_REOPERATOR(||)
};

struct capture
{
    template <class T>
    auto operator->*(const T& x) const
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

using string_map = std::unordered_map<std::string, std::vector<std::string>>;

template <class Keyword>
string_map parse(std::vector<std::string> as, Keyword keyword)
{
    string_map result;

    std::string flag;
    for(auto&& x : as)
    {
        auto f = keyword(x);
        if(f.empty())
        {
            result[flag].push_back(x);
        }
        else
        {
            flag = f.front();
            result[flag]; // Ensure the flag exists
        }
    }
    return result;
}

inline auto& get_test_cases()
{
    static std::vector<std::pair<std::string, std::function<void()>>> cases;
    return cases;
}

inline void add_test_case(std::string name, std::function<void()> f)
{
    get_test_cases().emplace_back(std::move(name), std::move(f));
}

struct auto_register_test_case
{
    template <class F>
    auto_register_test_case(const char* name, F f) noexcept
    {
        add_test_case(name, f);
    }
};

inline void run_test_case(const std::string& name, const std::function<void()>& f)
{
    std::cout << "[   RUN    ] " << name << std::endl;
    f();
    std::cout << "[ COMPLETE ] " << name << std::endl;
}

inline void run(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);

    auto args  = parse(as, [](auto &&) -> std::vector<std::string> { return {}; });
    auto cases = args[""];
    if(cases.empty())
    {
        for(auto&& tc : get_test_cases())
            run_test_case(tc.first, tc.second);
    }
    else
    {
        std::unordered_map<std::string, std::function<void()>> m(get_test_cases().begin(),
                                                                 get_test_cases().end());
        for(auto&& name : cases)
        {
            auto f = m.find(name);
            if(f == m.end())
                std::cout << "[  ERROR   ] Test case '" << name << "' not found." << std::endl;
            else
                run_test_case(name, f->second);
        }
    }
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

// NOLINTNEXTLINE
#define TEST_CAT(x, ...) TEST_PRIMITIVE_CAT(x, __VA_ARGS__)
// NOLINTNEXTLINE
#define TEST_PRIMITIVE_CAT(x, ...) x##__VA_ARGS__

// NOLINTNEXTLINE
#define TEST_CASE_REGISTER(...)                                                    \
    static test::auto_register_test_case TEST_CAT(register_test_case_, __LINE__) = \
        test::auto_register_test_case(#__VA_ARGS__, &__VA_ARGS__);

// NOLINTNEXTLINE
#define TEST_CASE(...)              \
    void __VA_ARGS__();             \
    TEST_CASE_REGISTER(__VA_ARGS__) \
    void __VA_ARGS__()

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

#endif
