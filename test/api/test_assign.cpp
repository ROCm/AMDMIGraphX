#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(shape_assign)
{
    auto s1_cpp = migraphx::shape{migraphx_shape_float_type, {1, 3}};
    std::vector<size_t> lens{2, 3};

    // handle ptr is const, workaround to construct shape using C API
    migraphx_shape_t s2;
    migraphx_shape_create(&s2, migraphx_shape_float_type, lens.data(), lens.size());
    auto s2_cpp = migraphx::shape(s2, migraphx::own{});
    CHECK(bool{s1_cpp != s2_cpp});
    // use C++ API for assignment
    s1_cpp.assign_to_handle(s2);
    CHECK(bool{s1_cpp == s2_cpp});

    auto s3_cpp = migraphx::shape{migraphx_shape_float_type, lens};
    // use C API for assignment
    migraphx_shape_assign_to(s2, s3_cpp.get_handle_ptr());
    CHECK(bool{s2_cpp == s3_cpp});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
