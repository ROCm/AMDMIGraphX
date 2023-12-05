#include <type_traits_test.hpp>

struct c1 {};
    
struct c2 {};
  
struct c3 : c2 {};
struct c1c2 {
    c1c2() {}
    c1c2(c1 const&) {}
    c1c2(c2 const&) {}
    c1c2& operator=(c1c2 const&) {
        return *this;
    }
};

#define CHECK_COMMON_TYPE(expected, ...) \
    CHECK_TYPE(rocm::common_type<__VA_ARGS__>::type, expected); \
    CHECK_TYPE(rocm::common_type_t<__VA_ARGS__>, expected);

#define CHECK_COMMON_TYPE2(expected, a, b) \
    CHECK_COMMON_TYPE(expected, a, b); \
    CHECK_COMMON_TYPE(expected, b, a);


DUAL_TEST_CASE()
{
    CHECK_COMMON_TYPE(int, int);
    CHECK_COMMON_TYPE(int, int, int);
    CHECK_COMMON_TYPE(unsigned int, unsigned int, unsigned int);
    CHECK_COMMON_TYPE(double, double, double);

    CHECK_COMMON_TYPE2(c1c2, c1c2&, c1&);
    CHECK_COMMON_TYPE2(c2*, c3*, c2*);
    CHECK_COMMON_TYPE2(const int*, int*, const int*);
    CHECK_COMMON_TYPE2(const volatile int*, volatile int*, const int*);
    CHECK_COMMON_TYPE2(volatile int*, int*, volatile int*);
    CHECK_COMMON_TYPE2(int, char, unsigned char);
    CHECK_COMMON_TYPE2(int, char, short);
    CHECK_COMMON_TYPE2(int, char, unsigned short);
    CHECK_COMMON_TYPE2(int, char, int);
    CHECK_COMMON_TYPE2(unsigned int, char, unsigned int);
    CHECK_COMMON_TYPE2(double, double, unsigned int);

    CHECK_COMMON_TYPE(double, double, char, int);

    CHECK_COMMON_TYPE(int, int&);
    CHECK_COMMON_TYPE(int, const int);
    CHECK_COMMON_TYPE(int, const int, const int);
    CHECK_COMMON_TYPE2(long, const int, const long);
}
