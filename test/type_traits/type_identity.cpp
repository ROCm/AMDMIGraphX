#include <type_traits_test.hpp>

DUAL_TEST_CASE()
{
    TRANSFORM_CHECK(rocm::type_identity,,);
    TRANSFORM_CHECK(rocm::type_identity, const, const);
    TRANSFORM_CHECK(rocm::type_identity, volatile, volatile);
    TRANSFORM_CHECK(rocm::type_identity, const volatile, const volatile);
    TRANSFORM_CHECK(rocm::type_identity, [], []);
    TRANSFORM_CHECK(rocm::type_identity, *const, *const);
    TRANSFORM_CHECK(rocm::type_identity, *volatile, *volatile);
    TRANSFORM_CHECK(rocm::type_identity, *const volatile, *const volatile);
    TRANSFORM_CHECK(rocm::type_identity, *, *);
    TRANSFORM_CHECK(rocm::type_identity, *, *);
    TRANSFORM_CHECK(rocm::type_identity, volatile*, volatile*);
    TRANSFORM_CHECK(rocm::type_identity, const[2], const[2]);
    TRANSFORM_CHECK(rocm::type_identity, volatile[2], volatile[2]);
    TRANSFORM_CHECK(rocm::type_identity, const volatile[2], const volatile[2]);
    TRANSFORM_CHECK(rocm::type_identity, [2], [2]);
    TRANSFORM_CHECK(rocm::type_identity, const*, const*);
    TRANSFORM_CHECK(rocm::type_identity, const*volatile, const*volatile);
    TRANSFORM_CHECK(rocm::type_identity, (), ());
    TRANSFORM_CHECK(rocm::type_identity, (int), (int));
    TRANSFORM_CHECK(rocm::type_identity, (*const)(), (*const)());
}
