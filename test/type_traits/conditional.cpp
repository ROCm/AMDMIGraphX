#include <type_traits_test.hpp>


ROCM_DUAL_TEST_CASE()
{
    static_assert(rocm::is_same<rocm::conditional<true, int, long>::type, int>{});
    static_assert(rocm::is_same<rocm::conditional<false, int, long>::type, long>{});
    static_assert(not rocm::is_same<rocm::conditional<true, int, long>::type, long>{});
    static_assert(not rocm::is_same<rocm::conditional<false, int, long>::type, int>{});

}