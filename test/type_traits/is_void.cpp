#include <type_traits_test.hpp>

DUAL_TEST_CASE()
{
    static_assert(rocm::is_void<void>{});
    static_assert(rocm::is_void<void const>{});
    static_assert(rocm::is_void<void volatile>{});
    static_assert(rocm::is_void<void const volatile>{});

    static_assert(not rocm::is_void<void*>{});
    static_assert(not rocm::is_void<int>{});
    static_assert(not rocm::is_void<test_tt::f1>{});
    static_assert(not rocm::is_void<test_tt::foo0_t>{});
    static_assert(not rocm::is_void<test_tt::foo1_t>{});
    static_assert(not rocm::is_void<test_tt::foo2_t>{});
    static_assert(not rocm::is_void<test_tt::foo3_t>{});
    static_assert(not rocm::is_void<test_tt::foo4_t>{});
    static_assert(not rocm::is_void<test_tt::incomplete_type>{});
    static_assert(not rocm::is_void<int&>{});
    static_assert(not rocm::is_void<int&&>{});

}
