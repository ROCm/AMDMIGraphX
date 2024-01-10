#include <type_traits_test.hpp>

ROCM_DUAL_TEST_CASE()
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
