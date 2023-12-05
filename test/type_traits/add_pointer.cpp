#include <type_traits_test.hpp>

DUAL_TEST_CASE()
{
    TRANSFORM_CHECK(rocm::add_pointer,,*);
    TRANSFORM_CHECK(rocm::add_pointer, const, const*);
    TRANSFORM_CHECK(rocm::add_pointer, volatile, volatile*);
    TRANSFORM_CHECK(rocm::add_pointer, *, **);
    TRANSFORM_CHECK(rocm::add_pointer, *volatile, *volatile*);
    TRANSFORM_CHECK(rocm::add_pointer, const*, const**);
    TRANSFORM_CHECK(rocm::add_pointer, volatile*, volatile**);
}
