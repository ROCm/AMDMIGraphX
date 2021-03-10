#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_tf)
{
    auto p      = migraphx::parse_tf("add_test.pb");
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
}

TEST_CASE(load_tf_multi)
{
    migraphx::tf_options tf_options;
    tf_options.set_output_names({"relu", "tanh"});
    auto p      = migraphx::parse_tf("multi_output_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
