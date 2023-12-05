#include <type_traits_test.hpp>

DUAL_TEST_CASE()
{
    TRANSFORM_CHECK(rocm::remove_reference,,)
    TRANSFORM_CHECK(rocm::remove_reference, &,)
    TRANSFORM_CHECK(rocm::remove_reference, &&,)
    TRANSFORM_CHECK(rocm::remove_reference, const, const)
    TRANSFORM_CHECK(rocm::remove_reference, volatile, volatile)
    TRANSFORM_CHECK(rocm::remove_reference, const &, const)
    TRANSFORM_CHECK(rocm::remove_reference, *, *)
    TRANSFORM_CHECK(rocm::remove_reference, *volatile, *volatile)
    TRANSFORM_CHECK(rocm::remove_reference, const &, const)
    TRANSFORM_CHECK(rocm::remove_reference, const*, const*)
    TRANSFORM_CHECK(rocm::remove_reference, volatile*, volatile*)
    TRANSFORM_CHECK(rocm::remove_reference, const[2], const[2])
    TRANSFORM_CHECK(rocm::remove_reference, (&)[2], [2])
    TRANSFORM_CHECK(rocm::remove_reference, const &&, const)
    TRANSFORM_CHECK(rocm::remove_reference, const &&, const)
    TRANSFORM_CHECK(rocm::remove_reference, (&&)[2], [2])
}
