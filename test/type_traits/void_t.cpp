#include <type_traits_test.hpp>

DUAL_TEST_CASE()
{
    CHECK_TYPE(rocm::void_t<int>, void);
    CHECK_TYPE(rocm::void_t<const volatile int>, void);
    CHECK_TYPE(rocm::void_t<int&>, void);
    CHECK_TYPE(rocm::void_t<void>, void);
    CHECK_TYPE(rocm::void_t<int(*)(int)>, void);
    CHECK_TYPE(rocm::void_t<int[]>, void);
    CHECK_TYPE(rocm::void_t<int[1]>, void);

    CHECK_TYPE(rocm::void_t<>, void);
    CHECK_TYPE(rocm::void_t<int, int>, void);

}
