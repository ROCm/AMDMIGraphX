#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(add_op)
{
    auto add_op = migraphx::operation("add");
    EXPECT(add_op.name() == "add");
}

TEST_CASE(reduce_mean_without_quotes)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean)
{
    auto rm = migraphx::operation("reduce_mean", "{\"axes\" : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean_with_format)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [%i, %i, %i, %i]}", 1, 2, 3, 4);
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(allocate_with_nested_json_str)
{
    auto alloc = migraphx::operation(
        "allocate", R"({"shape":{"type":"float_type","lens":[3, 3], "strides":[3, 1]}})"
);
    EXPECT(alloc.name() == "allocate");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
