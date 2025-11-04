#ifndef MIGRAPHX_GUARD_KERNELS_TEST_HPP
#define MIGRAPHX_GUARD_KERNELS_TEST_HPP

#include <migraphx/kernels/print.hpp>
#include <migraphx/kernels/hip.hpp>

namespace migraphx {

namespace test {

template <int N>
struct rank : rank<N - 1>
{
};

template <>
struct rank<0>
{
};

// clang-format off
// NOLINTNEXTLINE
#define TEST_FOREACH_BINARY_OPERATORS(m) \
    m(==, equal) \
    m(!=, not_equal) \
    m(<=, less_than_equal) \
    m(>=, greater_than_equal) \
    m(<, less_than) \
    m(>, greater_than) \
    m(and, and_op) \
    m(or, or_op)
// clang-format on

// clang-format off
// NOLINTNEXTLINE
#define TEST_FOREACH_UNARY_OPERATORS(m) \
    m(not, not_op)
// clang-format on

// NOLINTNEXTLINE
#define TEST_EACH_BINARY_OPERATOR_OBJECT(op, name)               \
    struct name                                                  \
    {                                                            \
        static constexpr const char* as_string() { return #op; } \
        template <class T, class U>                              \
        static constexpr decltype(auto) call(T && x, U&& y)      \
        {                                                        \
            return x op y;                                       \
        }                                                        \
    };

// NOLINTNEXTLINE
#define TEST_EACH_UNARY_OPERATOR_OBJECT(op, name)                \
    struct name                                                  \
    {                                                            \
        static constexpr const char* as_string() { return #op; } \
        template <class T>                                       \
        static constexpr decltype(auto) call(T && x)             \
        {                                                        \
            return op x;                                         \
        }                                                        \
    };

TEST_FOREACH_BINARY_OPERATORS(TEST_EACH_BINARY_OPERATOR_OBJECT)
TEST_FOREACH_UNARY_OPERATORS(TEST_EACH_UNARY_OPERATOR_OBJECT)

struct nop
{
    static constexpr const char* as_string() { return ""; }
    template <class T>
    static constexpr auto call(T&& x)
    {
        return static_cast<T&&>(x);
    }
};

struct function
{
    static constexpr const char* as_string() { return ""; }
    template <class T>
    static constexpr decltype(auto) call(T&& x)
    {
        return x();
    }
};

template <class Stream, class T>
constexpr Stream& print_stream_impl(rank<0>, Stream& s, const T&)
{
    // TODO: Print typename
    s << '?';
    return s;
}

template <class Stream, class T>
constexpr auto print_stream_impl(rank<1>, Stream& s, const T& x) -> decltype(s << x)
{
    return s << x;
}

template <class Stream, class T>
constexpr void print_stream(Stream& s, const T& x)
{
    print_stream_impl(rank<2>{}, s, x);
}

template <class T>
constexpr const T& get_value(const T& x)
{
    return x;
}

template <class T, class Operator = nop>
struct lhs_expression;

template <class T>
constexpr lhs_expression<T> make_lhs_expression(T&& lhs);

template <class T, class Operator>
constexpr lhs_expression<T, Operator> make_lhs_expression(T&& lhs, Operator);

// NOLINTNEXTLINE
#define TEST_EXPR_BINARY_OPERATOR(op, name)                                         \
    template <class V>                                                              \
    constexpr auto operator op(V&& rhs2) const                                      \
    {                                                                               \
        return make_expression(*this, static_cast<V&&>(rhs2), name{}); /* NOLINT */ \
    }

// NOLINTNEXTLINE
#define TEST_EXPR_UNARY_OPERATOR(op, name) \
    constexpr auto operator op() const { return make_lhs_expression(lhs, name{}); /* NOLINT */ }

template <class T, class U, class Operator>
struct expression
{
    T lhs;
    U rhs;

    template <class Stream>
    friend constexpr Stream& operator<<(Stream& s, const expression& self)
    {
        print_stream(s, self.lhs);
        s << " " << Operator::as_string() << " ";
        print_stream(s, self.rhs);
        return s;
    }

    friend constexpr decltype(auto) get_value(const expression& e) { return e.value(); }

    constexpr decltype(auto) value() const
    {
        return Operator::call(get_value(lhs), get_value(rhs));
    };

    TEST_FOREACH_UNARY_OPERATORS(TEST_EXPR_UNARY_OPERATOR)
    TEST_FOREACH_BINARY_OPERATORS(TEST_EXPR_BINARY_OPERATOR)
};

// TODO: Remove rvalue references
template <class T, class U, class Operator>
constexpr expression<T, U, Operator> make_expression(T&& lhs, U&& rhs, Operator)
{
    return {static_cast<T&&>(lhs), static_cast<U&&>(rhs)};
}

// TODO: Remove rvalue reference
template <class T>
constexpr lhs_expression<T> make_lhs_expression(T&& lhs)
{
    return lhs_expression<T>{static_cast<T&&>(lhs)};
}

template <class T, class Operator>
constexpr lhs_expression<T, Operator> make_lhs_expression(T&& lhs, Operator)
{
    return lhs_expression<T, Operator>{static_cast<T&&>(lhs)};
}

template <class T, class Operator>
struct lhs_expression
{
    T lhs;
    constexpr explicit lhs_expression(T e) : lhs(static_cast<T&&>(e)) {}

    template <class Stream>
    friend constexpr Stream& operator<<(Stream& s, const lhs_expression& self)
    {
        const char* op = Operator::as_string();
        if(op != nullptr and *op != '\0')
            s << Operator::as_string() << " ";
        print_stream(s, self.lhs);
        return s;
    }

    friend constexpr decltype(auto) get_value(const lhs_expression& e) { return e.value(); }

    constexpr decltype(auto) value() const { return Operator::call(get_value(lhs)); }

    TEST_FOREACH_BINARY_OPERATORS(TEST_EXPR_BINARY_OPERATOR)
    TEST_FOREACH_UNARY_OPERATORS(TEST_EXPR_UNARY_OPERATOR)

// NOLINTNEXTLINE
#define TEST_LHS_REOPERATOR(op)                    \
    template <class U>                             \
    constexpr auto operator op(const U& rhs) const \
    {                                              \
        return make_lhs_expression(lhs op rhs);    \
    }
    TEST_LHS_REOPERATOR(+)
    TEST_LHS_REOPERATOR(-)
    TEST_LHS_REOPERATOR(*)
    TEST_LHS_REOPERATOR(/)
    TEST_LHS_REOPERATOR(%)
    TEST_LHS_REOPERATOR(&)
    TEST_LHS_REOPERATOR(|)
    TEST_LHS_REOPERATOR(^)
};

struct capture
{
    template <class T>
    constexpr auto operator->*(const T& x) const
    {
        return make_lhs_expression(x);
    }

    template <class T, class Operator>
    constexpr auto operator->*(const lhs_expression<T, Operator>& x) const
    {
        return x;
    }
};

struct test_manager
{
    int32_t* failures = nullptr;

    __device__ void report_failure() const { (*failures)++; }

    template <class T, class F>
    __device__ void
    failed(const T& x, const char* msg, const char* func, const char* file, int line, F f)
    {
        // TODO: Check failures across multiple lanes
        if(not bool(x.value()))
        {
            // NOLINTNEXTLINE(readability-static-accessed-through-instance)
            if(threadIdx.x == 0)
            {
                migraphx::cout() << func << '\n';
                migraphx::cout() << file << ':' << line << ':' << '\n';
                migraphx::cout() << "    FAILED: " << msg << " [ " << x << " ]" << '\n';
                report_failure();
            }
            f();
        }
    }
};

[[noreturn]] __device__ inline void fail()
{
    // There is no way to easily exit with no error. We can terminate the
    // current wavefront without an error, but if there is more wavefronts
    // than we need to fallback to a trap which throws an error in HSA
    // runtime unfortunately.
    auto nb = gridDim.x * gridDim.y * gridDim.z; // NOLINT(readability-static-accessed-through-instance)
    auto bs = blockDim.x * blockDim.y * blockDim.z; // NOLINT(readability-static-accessed-through-instance)
    if(nb == 1 and bs <= __builtin_amdgcn_wavefrontsize())
    {
        __builtin_amdgcn_endpgm();
    }
    else
    {
        __builtin_trap();
    }
}

#ifdef CPPCHECK
// NOLINTNEXTLINE
#define TEST_CAPTURE(...) __VA_ARGS__
#else
// NOLINTNEXTLINE
#define TEST_CAPTURE(...) migraphx::test::capture{}->*__VA_ARGS__
#endif

#ifdef _WIN32
// NOLINTNEXTLINE
#define TEST_PRETTY_FUNCTION __FUNCSIG__
#else
// NOLINTNEXTLINE
#define TEST_PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

// NOLINTNEXTLINE
#define CHECK(...)                        \
    migraphx_private_test_manager.failed( \
        TEST_CAPTURE(__VA_ARGS__), #__VA_ARGS__, TEST_PRETTY_FUNCTION, __FILE__, __LINE__, [] {})

// NOLINTNEXTLINE
#define EXPECT(...)                                                 \
    migraphx_private_test_manager.failed(TEST_CAPTURE(__VA_ARGS__), \
                                         #__VA_ARGS__,              \
                                         TEST_PRETTY_FUNCTION,      \
                                         __FILE__,                  \
                                         __LINE__,                  \
                                         &migraphx::test::fail)

// NOLINTNEXTLINE
#define TEST_CASE(...)                                   \
    __device__ [[maybe_unused]] static void __VA_ARGS__( \
        [[maybe_unused]] migraphx::test::test_manager& migraphx_private_test_manager)

} // namespace test
} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TEST_HPP
