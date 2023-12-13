
#include <onnx_test.hpp>

TEST_CASE(implicit_add_bcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {3, 4, 1}});
    auto l3 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), l1);
    mm->add_instruction(migraphx::make_op("add"), l0, l3);

    auto prog = optimize_onnx("implicit_add_bcast_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(implicit_add_bcast_user_input_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {4, 5, 1}});
    auto l3 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 4, 5, 6}}}), l1);
    auto r = mm->add_instruction(migraphx::make_op("add"), l0, l3);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_input_dims["0"] = {3, 4, 5, 6};
    options.map_input_dims["1"] = {4, 5, 1};
    auto prog                   = migraphx::parse_onnx("implicit_add_bcast_test.onnx", options);

    EXPECT(p == prog);
}
