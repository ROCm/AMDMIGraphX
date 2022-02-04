#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(shape_assign)
{
    auto s1 = migraphx::shape{migraphx_shape_float_type, {1, 3}};
    migraphx_shape_t s2;
    std::vector<size_t> lens{2, 3};
    // handle ptr is const, workaround to construct shape using C API
    migraphx_shape_create(&s2, migraphx_shape_float_type, lens.data(), lens.size());
    auto s2_cpp = migraphx::shape(s2, migraphx::own{});
    CHECK(bool{s1 != s2_cpp});
    migraphx_shape_assign(s2, s1.get_handle_ptr());
    CHECK(bool{s1 == s2_cpp});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
