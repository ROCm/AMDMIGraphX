#include <type_traits_test.hpp>

DUAL_TEST_CASE()
{
    TRANSFORM_CHECK(rocm::remove_cv,,);
    TRANSFORM_CHECK(rocm::remove_cv, const,);
    TRANSFORM_CHECK(rocm::remove_cv, volatile,);
    TRANSFORM_CHECK(rocm::remove_cv, const volatile,);
    TRANSFORM_CHECK(rocm::remove_cv, const &, const&);
    TRANSFORM_CHECK(rocm::remove_cv, *const, *);
    TRANSFORM_CHECK(rocm::remove_cv, *volatile, *);
    TRANSFORM_CHECK(rocm::remove_cv, *const volatile, *);
    TRANSFORM_CHECK(rocm::remove_cv, *, *);
    TRANSFORM_CHECK(rocm::remove_cv, const*, const*);
    TRANSFORM_CHECK(rocm::remove_cv, volatile*, volatile*);
    TRANSFORM_CHECK(rocm::remove_cv, const[2], [2]);
    TRANSFORM_CHECK(rocm::remove_cv, volatile[2], [2]);
    TRANSFORM_CHECK(rocm::remove_cv, const volatile[2], [2]);
    TRANSFORM_CHECK(rocm::remove_cv, [2], [2]);
    TRANSFORM_CHECK(rocm::remove_cv, const*, const*);
    TRANSFORM_CHECK(rocm::remove_cv, const*volatile, const*);
    TRANSFORM_CHECK(rocm::remove_cv, const &&, const&&);
}
