#include <type_traits_test.hpp>

ROCM_DUAL_TEST_CASE()
{
    ROCM_TRANSFORM_CHECK(rocm::add_pointer,,*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, const, const*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, volatile, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, *, **);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, *volatile, *volatile*);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, const*, const**);
    ROCM_TRANSFORM_CHECK(rocm::add_pointer, volatile*, volatile**);
}
