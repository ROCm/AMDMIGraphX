
#include <onnx_test.hpp>

TEST_CASE(variable_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_onnx("variable_batch_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 2};

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{2, 5}, {3, 3}, {16, 16}, {16, 16}}});
    auto r = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 5};

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{2, 5}, {3, 3}, {16, 16}, {16, 16}}});
    auto r = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 5}, {3, 3}, {16, 16}, {16, 16}};

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dim_value = 2;

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test5)
{
    // Error using default_dim_value and default_dyn_dim_value
    migraphx::onnx_options options;
    options.default_dim_value     = 2;
    options.default_dyn_dim_value = {1, 2};

    EXPECT(test::throws([&] { migraphx::parse_onnx("variable_batch_test.onnx", options); }));
}

TEST_CASE(variable_batch_user_input_test6)
{
    // Error using both map_dyn_input_dims and map_input_dims
    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 5}, {3, 3}, {16, 16}, {16, 16}};
    options.map_input_dims["0"]     = {2, 3, 16, 16};

    EXPECT(test::throws([&] { migraphx::parse_onnx("variable_batch_test.onnx", options); }));
}

TEST_CASE(variable_batch_user_input_test7)
{
    // if entry in map_dyn_input_dims is all fixed dynamic_dimensions, convert it to a static
    // shape
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 2, {2}}, {3, 3}, {16, 16}, {16, 16}};

    auto prog = migraphx::parse_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}
