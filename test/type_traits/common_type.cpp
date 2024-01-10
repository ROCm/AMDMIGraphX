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

#define ROCM_CHECK_COMMON_TYPE(expected, ...) \
    ROCM_CHECK_TYPE(rocm::common_type<__VA_ARGS__>::type, expected); \
    ROCM_CHECK_TYPE(rocm::common_type_t<__VA_ARGS__>, expected);

#define ROCM_CHECK_COMMON_TYP_E2(expected, a, b) \
    ROCM_CHECK_COMMON_TYPE(expected, a, b); \
    ROCM_CHECK_COMMON_TYPE(expected, b, a);


ROCM_DUAL_TEST_CASE()
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
