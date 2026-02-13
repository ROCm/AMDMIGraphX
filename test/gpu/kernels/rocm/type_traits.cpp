
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

namespace test_tt {

template <class T, class U>
struct is_same
{
    constexpr operator bool() const noexcept { return false; }
};

template <class T>
struct is_same<T, T>
{
    constexpr operator bool() const noexcept { return true; }
};

#define ROCM_CHECK_TYPE(...) EXPECT(test_tt::is_same<__VA_ARGS__>{})

enum enum1
{
    enum1_one,
    enum1_two
};

struct udt
{
    udt();
    ~udt();
    udt(const udt&);
    udt& operator=(const udt&);
    int i;

    void f1();
    int f2();
    int f3(int);
    int f4(int, float);
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wredundant-parens"
using f1  = void (*)();
using f2  = int (*)(int);
using f3  = int (*)(int, bool);
using mf1 = void (udt::*)();
using mf2 = int (udt::*)();
using mf3 = int (udt::*)(int);
using mf4 = int (udt::*)(int, float);
using mp  = int(udt::*);
using cmf = int (udt::*)(int) const;
using mf8 = int (udt::*)(...);
#pragma clang diagnostic pop

using foo0_t = void();
using foo1_t = void(int);
using foo2_t = void(int&, double);
using foo3_t = void(int&, bool, int, int);
using foo4_t = void(int, bool, int*, int*, int, int, int, int, int);

struct incomplete_type;

// clang-format off
#define ROCM_VISIT_TYPES(m, ...) \
    m(bool, __VA_ARGS__) \
    m(char, __VA_ARGS__) \
    m(wchar_t, __VA_ARGS__) \
    m(signed char, __VA_ARGS__) \
    m(unsigned char, __VA_ARGS__) \
    m(short, __VA_ARGS__) \
    m(unsigned short, __VA_ARGS__) \
    m(int, __VA_ARGS__) \
    m(unsigned int, __VA_ARGS__) \
    m(long, __VA_ARGS__) \
    m(unsigned long, __VA_ARGS__) \
    m(float, __VA_ARGS__) \
    m(long double, __VA_ARGS__) \
    m(double, __VA_ARGS__) \
    m(test_tt::udt, __VA_ARGS__) \
    m(test_tt::enum1, __VA_ARGS__)
// clang-format on

#define ROCM_TRANSFORM_CHECK_VISITOR(x, name, from_suffix, to_suffix) \
    ROCM_CHECK_TYPE(x to_suffix, name<x from_suffix>::type);          \
    ROCM_CHECK_TYPE(x to_suffix, name##_t<x from_suffix>);

#define ROCM_TRANSFORM_CHECK(name, from_suffix, to_suffix) \
    ROCM_VISIT_TYPES(ROCM_TRANSFORM_CHECK_VISITOR, name, from_suffix, to_suffix)

} // namespace test_tt

TEST_CASE(add_pointer)
{
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, , *);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, const, const*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, volatile, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, *, **);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, * volatile, * volatile*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, const*, const**);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, volatile*, volatile**);
}

TEST_CASE(remove_pointer)
{
    // Non-pointer types are unchanged
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, , );
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, const, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, volatile, volatile);
    // Pointer types are stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, *, );
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, * const, );
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, * volatile, );
    // Pointer to cv-qualified types
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, const*, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, volatile*, volatile);
    // References are unchanged
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, &, &);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, const&, const&);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, &&, &&);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, const&&, const&&);
    // Arrays are unchanged
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, const[2], const[2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, const[2][3], const[2][3]);
    // References to arrays are unchanged
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, (&)[2], (&)[2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, (&&)[2], (&&)[2]);
    // Function pointers are stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, (*)(long), (long));
    ROCM_TRANSFORM_CHECK(rocm::remove_pointer, (*const)(long), (long));
}

struct c1
{
};

struct c2
{
};

struct c3 : c2
{
};
struct c1c2
{
    c1c2() {}
    c1c2(c1 const&) {}
    c1c2(c2 const&) {}
    c1c2& operator=(c1c2 const&) { return *this; }
};

#define ROCM_CHECK_COMMON_TYPE(expected, ...)                        \
    ROCM_CHECK_TYPE(rocm::common_type<__VA_ARGS__>::type, expected); \
    ROCM_CHECK_TYPE(rocm::common_type_t<__VA_ARGS__>, expected);

#define ROCM_CHECK_COMMON_TYP_E2(expected, a, b) \
    ROCM_CHECK_COMMON_TYPE(expected, a, b);      \
    ROCM_CHECK_COMMON_TYPE(expected, b, a);

TEST_CASE(common_type)
{
    ROCM_CHECK_COMMON_TYPE(int, int);
    ROCM_CHECK_COMMON_TYPE(int, int, int);
    ROCM_CHECK_COMMON_TYPE(unsigned int, unsigned int, unsigned int);
    ROCM_CHECK_COMMON_TYPE(double, double, double);

    ROCM_CHECK_COMMON_TYP_E2(c1c2, c1c2&, c1&);
    ROCM_CHECK_COMMON_TYP_E2(c2*, c3*, c2*);
    ROCM_CHECK_COMMON_TYP_E2(const int*, int*, const int*);
    ROCM_CHECK_COMMON_TYP_E2(const volatile int*, volatile int*, const int*);
    ROCM_CHECK_COMMON_TYP_E2(volatile int*, int*, volatile int*);
    ROCM_CHECK_COMMON_TYP_E2(int, char, unsigned char);
    ROCM_CHECK_COMMON_TYP_E2(int, char, short);
    ROCM_CHECK_COMMON_TYP_E2(int, char, unsigned short);
    ROCM_CHECK_COMMON_TYP_E2(int, char, int);
    ROCM_CHECK_COMMON_TYP_E2(unsigned int, char, unsigned int);
    ROCM_CHECK_COMMON_TYP_E2(double, double, unsigned int);

    ROCM_CHECK_COMMON_TYPE(double, double, char, int);

    ROCM_CHECK_COMMON_TYPE(int, int&);
    ROCM_CHECK_COMMON_TYPE(int, const int);
    ROCM_CHECK_COMMON_TYPE(int, const int, const int);
    ROCM_CHECK_COMMON_TYP_E2(long, const int, const long);
}

TEST_CASE(condition)
{
    EXPECT(rocm::is_same<rocm::conditional<true, int, long>::type, int>{});
    EXPECT(rocm::is_same<rocm::conditional<false, int, long>::type, long>{});
    EXPECT(not rocm::is_same<rocm::conditional<true, int, long>::type, long>{});
    EXPECT(not rocm::is_same<rocm::conditional<false, int, long>::type, int>{});
}

TEST_CASE(is_void)
{
    EXPECT(rocm::is_void<void>{});
    EXPECT(rocm::is_void<void const>{});
    EXPECT(rocm::is_void<void volatile>{});
    EXPECT(rocm::is_void<void const volatile>{});

    EXPECT(not rocm::is_void<void*>{});
    EXPECT(not rocm::is_void<int>{});
    EXPECT(not rocm::is_void<test_tt::f1>{});
    EXPECT(not rocm::is_void<test_tt::foo0_t>{});
    EXPECT(not rocm::is_void<test_tt::foo1_t>{});
    EXPECT(not rocm::is_void<test_tt::foo2_t>{});
    EXPECT(not rocm::is_void<test_tt::foo3_t>{});
    EXPECT(not rocm::is_void<test_tt::foo4_t>{});
    EXPECT(not rocm::is_void<test_tt::incomplete_type>{});
    EXPECT(not rocm::is_void<int&>{});
    EXPECT(not rocm::is_void<int&&>{});
}

TEST_CASE(remove_cv)
{
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, , );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, volatile, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const volatile, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const&, const&);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, * const, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, * volatile, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, * const volatile, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, *, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, volatile*, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, volatile[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const volatile[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, [2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const* volatile, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const&&, const&&);
}

TEST_CASE(remove_reference)
{
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, , );
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, &, );
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, &&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, volatile, volatile);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, *, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, * volatile, * volatile);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, volatile*, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const[2], const[2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, (&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, (&&)[2], [2]);
}

TEST_CASE(remove_cvref)
{
    // cv-qualifiers stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, , );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const volatile, );
    // lvalue references with cv-qualifiers stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, &, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const volatile&, );
    // rvalue references with cv-qualifiers stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, &&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const&&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile&&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const volatile&&, );
    // Pointer top-level cv stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, *, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * const, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * volatile, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * const volatile, *);
    // Pointer top-level cv stripped via lvalue ref
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * &, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * const&, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * volatile&, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * const volatile&, *);
    // Pointer top-level cv stripped via rvalue ref
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * &&, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * const&&, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * volatile&&, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, * const volatile&&, *);
    // Pointee cv-qualifiers preserved
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile*, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* const, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* volatile, const*);
    // Pointee cv-qualifiers preserved via lvalue ref
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* &, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile* &, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* const&, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* volatile&, const*);
    // Pointee cv-qualifiers preserved via rvalue ref
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* &&, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile* &&, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* const&&, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const* volatile&&, const*);
    // Arrays with cv-qualifiers stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, [2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const volatile[2], [2]);
    // Arrays via lvalue ref with cv-qualifiers stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, (&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const(&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile(&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const volatile(&)[2], [2]);
    // Arrays via rvalue ref with cv-qualifiers stripped
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, (&&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const(&&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, volatile(&&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cvref, const volatile(&&)[2], [2]);
}

TEST_CASE(type_identity)
{
    ROCM_TRANSFORM_CHECK(rocm::type_identity, , );
    ROCM_TRANSFORM_CHECK(rocm::type_identity, const, const);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, volatile, volatile);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, const volatile, const volatile);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, [], []);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, * const, * const);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, * volatile, * volatile);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, * const volatile, * const volatile);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, *, *);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, volatile*, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, const[2], const[2]);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, volatile[2], volatile[2]);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, const volatile[2], const volatile[2]);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, [2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, const* volatile, const* volatile);
    ROCM_TRANSFORM_CHECK(rocm::type_identity, (), ());
    ROCM_TRANSFORM_CHECK(rocm::type_identity, (int), (int));
    ROCM_TRANSFORM_CHECK(rocm::type_identity, (*const)(), (*const)());
}

TEST_CASE(void_t)
{
    ROCM_CHECK_TYPE(rocm::void_t<int>, void);
    ROCM_CHECK_TYPE(rocm::void_t<const volatile int>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int&>, void);
    ROCM_CHECK_TYPE(rocm::void_t<void>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int (*)(int)>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int[]>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int[1]>, void);

    ROCM_CHECK_TYPE(rocm::void_t<>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int, int>, void);
}
