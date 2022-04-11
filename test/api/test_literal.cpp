#include <cstdint>
#include <migraphx/migraphx.hpp>
#include <migraphx/migraphx.h>
#include "test.hpp"

TEST_CASE(create_literal)
{
    migraphx::shape literal_shape{migraphx_shape_float_type, {2, 3}};
    std::vector<float> literal_values{-3.14, -2.14, -1.14, 0, 1, 2.3};
    auto xl = migraphx::literal(literal_shape, reinterpret_cast<char*>(literal_values.data()));
    EXPECT(bool{xl.get_shape() == literal_shape});
    const auto* xl_value_ptr = reinterpret_cast<const float*>(xl.data());
    std::vector<float> xl_values(xl_value_ptr, xl_value_ptr + 6);
    EXPECT(bool{literal_values == xl_values});
}

TEST_CASE(literal_equal)
{
    migraphx::shape x_shape{migraphx_shape_int32_type, {1, 3}};
    std::vector<int32_t> x_values = {1, 2, 3};
    char* x_values_ptr            = reinterpret_cast<char*>(x_values.data());
    migraphx::literal x_cpp       = migraphx::literal{x_shape, x_values_ptr};

    migraphx_literal_t y;
    migraphx::shape y_shape{migraphx_shape_float_type, {1, 3}};
    std::vector<float> y_values = {1.1, 2.2, 3.3};
    char* y_values_ptr          = reinterpret_cast<char*>(y_values.data());
    migraphx_literal_create(&y, y_shape.get_handle_ptr(), y_values_ptr);

    auto y_cpp = migraphx::literal(y, migraphx::own{});
    EXPECT(bool{x_cpp != y_cpp});
    x_cpp.assign_to_handle(y);
    EXPECT(bool{x_cpp == y_cpp});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
