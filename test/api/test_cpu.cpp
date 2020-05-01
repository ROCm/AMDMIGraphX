#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_and_run)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("cpu"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(load_and_run_user_input_shape)
{
    migraphx::onnx_options options;
    options.set_input_parameter_shape("0", {2, 3, 64, 64});
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx", options);
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("cpu"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(zero_parameter)
{
    auto p             = migraphx::parse_onnx("constant_fill_test.onnx");
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("cpu"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
