#include <type_traits_test.hpp>

ROCM_DUAL_TEST_CASE()
{
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, , );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, volatile, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const volatile, );
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const&, const&);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, * const, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, * volatile, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, * const volatile, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, *, *);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, volatile*, volatile*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, volatile[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const volatile[2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, [2], [2]);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const*, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const* volatile, const*);
    ROCM_TRANSFORM_CHECK(rocm::remove_cv, const&&, const&&);
}
