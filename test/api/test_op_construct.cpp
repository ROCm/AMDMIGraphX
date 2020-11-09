#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(add_op)
{
    auto add_op = migraphx::operation("add");
    EXPECT(add_op.name() == "add");
}

TEST_CASE(reduce_mean)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean1)
{
    auto rm = migraphx::operation("reduce_mean", "{\"axes\" : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
