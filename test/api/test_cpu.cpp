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

TEST_CASE(quantize_fp16)
{
    auto p1        = migraphx::parse_onnx("gemm_ex_test.onnx");
    const auto& p2 = p1;
    const auto& p3 = p1;
    migraphx::quantize_fp16(p1);

    migraphx::quantize_op_names names;
    migraphx::quantize_fp16(p2, names);
    CHECK(bool{p1 == p2});

    names.add("dot");
    migraphx::quantize_fp16(p3, names);
    CHECK(bool{p1 == p3});
}

TEST_CASE(quantize_int8)
{
    auto p1        = migraphx::parse_onnx("gemm_ex_test.onnx");
    const auto& p2 = p1;
    auto t         = migraphx::target("cpu");
    migraphx::quantize_options options;
    migraphx::quantize_int8(p1, t, options);

    migraphx::program_parameters pp;
    auto param_shapes = p1.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    options.add_calibration_data(pp);
    options.add_op_name("dot");

    migraphx::quantize_int8(p2, t, options);
    CHECK(bool{p1 == p2});
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

TEST_CASE(set_scalar_parameter)
{
    auto p1 = migraphx::parse_onnx("add_bcast_test.onnx");
    migraphx::shape s1(migraphx_shape_float_type, {3, 4});
    auto param_shapes = p1.get_parameter_shapes();
    auto s1_orig      = param_shapes["1"];
    CHECK(bool{s1 == s1_orig});

    migraphx::onnx_options option;
    option.set_input_parameter_shape("1", {});
    auto p2 = migraphx::parse_onnx("add_bcast_test.onnx", option);
    migraphx::shape s_scalar(migraphx_shape_float_type);
    auto param_shapes_1 = p2.get_parameter_shapes();
    auto s_scalar_after = param_shapes_1["1"];
    CHECK(bool{s_scalar == s_scalar_after});
}

TEST_CASE(scalar_shape)
{
    auto s = migraphx::shape(migraphx_shape_float_type);
    EXPECT(s.lengths().size() == 1);
    EXPECT(s.strides().size() == 1);
    EXPECT(s.lengths().front() == 1);
    EXPECT(s.strides().front() == 0);
}

TEST_CASE(strided_shape)
{
    std::vector<std::size_t> lens    = {2, 2};
    std::vector<std::size_t> strides = {1, 2};
    auto s                           = migraphx::shape(migraphx_shape_float_type, lens, strides);
    EXPECT(s.lengths() == lens);
    EXPECT(s.strides() == strides);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
