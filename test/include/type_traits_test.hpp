
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

#define ROCM_CHECK_TYPE(...) static_assert(test_tt::is_same<__VA_ARGS__>{})


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

using f1 = void (*)();
using f2 = int (*)(int);
using f3 = int (*)(int, bool);
using mf1 = void (udt::*)();
using mf2 = int (udt::*)();
using mf3 = int (udt::*)(int);
using mf4 = int (udt::*)(int, float);
using mp = int (udt::*);
using cmf = int (udt::*)(int) const;
using mf8 = int (udt::*)(...);

using foo0_t = void ();
using foo1_t = void (int);
using foo2_t = void (int &, double);
using foo3_t = void (int &, bool, int, int);
using foo4_t = void (int, bool, int *, int *, int, int, int, int, int);

struct incomplete_type;

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

#define ROCM_TRANSFORM_CHECK_VISITOR(x, name, from_suffix, to_suffix) \
    ROCM_CHECK_TYPE(x to_suffix, name<x from_suffix>::type); \
    ROCM_CHECK_TYPE(x to_suffix, name##_t<x from_suffix>);

#define ROCM_TRANSFORM_CHECK(name, from_suffix, to_suffix) \
    ROCM_VISIT_TYPES(ROCM_TRANSFORM_CHECK_VISITOR, name, from_suffix, to_suffix)


} // namespace test_tt
