#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_tf)
{
    auto p      = migraphx::parse_tf("add_test.pb");
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
}

TEST_CASE(load_tf_default_dim)
{
    migraphx::tf_options tf_options;
    size_t batch = 2;
    tf_options.set_default_dim_value(batch);
    tf_options.set_nhwc();
    auto p      = migraphx::parse_tf("conv_batch_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
    CHECK(shapes.front().lengths().front() == batch);
}

TEST_CASE(load_tf_param_shape)
{
    migraphx::tf_options tf_options;
    std::vector<size_t> new_shape{1, 3};
    tf_options.set_input_parameter_shape("0", new_shape);
    tf_options.set_input_parameter_shape("1", new_shape);
    auto p      = migraphx::parse_tf("add_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
    CHECK(shapes.front().lengths() == new_shape);
}

TEST_CASE(load_tf_multi_outputs)
{
    migraphx::tf_options tf_options;
    tf_options.set_output_names({"relu", "tanh"});
    auto p      = migraphx::parse_tf("multi_output_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
