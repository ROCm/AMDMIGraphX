#include <type_traits_test.hpp>

ROCM_DUAL_TEST_CASE()
{
    ROCM_CHECK_TYPE(rocm::void_t<int>, void);
    ROCM_CHECK_TYPE(rocm::void_t<const volatile int>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int&>, void);
    ROCM_CHECK_TYPE(rocm::void_t<void>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int(*)(int)>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int[]>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int[1]>, void);

    ROCM_CHECK_TYPE(rocm::void_t<>, void);
    ROCM_CHECK_TYPE(rocm::void_t<int, int>, void);

}
