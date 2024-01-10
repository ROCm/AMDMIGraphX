#include <type_traits_test.hpp>

ROCM_DUAL_TEST_CASE()
{
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, , );
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, &, );
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, &&, );
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, volatile, volatile);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, *, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, * volatile, * volatile);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, volatile*, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const[2], const[2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, (&)[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, const&&, const);
    ROCM_TRANSFORM_CHECK(rocm::remove_reference, (&&)[2], [2]);
}
