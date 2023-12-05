
#include <rocm/type_traits.hpp>

#include <dual_test.hpp>

namespace test_tt {

template<class T, class U>
struct is_same {
    constexpr operator bool() const noexcept { return false; }
};
 
template<class T>
struct is_same<T, T> {
    constexpr operator bool() const noexcept { return true; }
};

#define CHECK_TYPE(...) static_assert(test_tt::is_same<__VA_ARGS__>{})


enum enum1
{
   enum1_one, enum1_two
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

typedef void(*f1)();
typedef int(*f2)(int);
typedef int(*f3)(int, bool);
typedef void (udt::*mf1)();
typedef int (udt::*mf2)();
typedef int (udt::*mf3)(int);
typedef int (udt::*mf4)(int, float);
typedef int (udt::*mp);
typedef int (udt::*cmf)(int) const;
typedef int (udt::*mf8)(...);

#define VISIT_TYPES(m, ...) \
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

#define TRANSFORM_CHECK_VISITOR(x, name, from_suffix, to_suffix) \
    CHECK_TYPE(x to_suffix, name<x from_suffix>::type); \
    CHECK_TYPE(x to_suffix, name##_t<x from_suffix>);

#define TRANSFORM_CHECK(name, from_suffix, to_suffix) \
    VISIT_TYPES(TRANSFORM_CHECK_VISITOR, name, from_suffix, to_suffix)


} // namespace test_tt
