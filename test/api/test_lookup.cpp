#include <migraphx/migraphx.hpp>
#include "test.hpp"

template <class T>
std::false_type has_handle(migraphx::rank<0>, T)
{
    return {};
}

template <class T>
auto has_handle(migraphx::rank<1>, T*) -> decltype(migraphx::as_handle<T>{}, std::true_type{})
{
    return {};
}

TEST_CASE(shape)
{
    static_assert(std::is_same<migraphx::as_handle<migraphx_shape>, migraphx::shape>{}, "Failed");
    static_assert(std::is_same<migraphx::as_handle<migraphx_shape_t>, migraphx::shape>{}, "Failed");
    static_assert(std::is_same<migraphx::as_handle<const_migraphx_shape_t>, migraphx::shape>{},
                  "Failed");
}
TEST_CASE(non_handle)
{
    int i = 0;
    EXPECT(bool{has_handle(migraphx::rank<1>{}, migraphx_shape_t{})});
    EXPECT(bool{not has_handle(migraphx::rank<1>{}, &i)});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
