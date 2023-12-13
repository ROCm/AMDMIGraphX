
#include <onnx_test.hpp>

TEST_CASE(equal_bool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sf{migraphx::shape::float_type, {2, 3}};
    migraphx::shape sb{migraphx::shape::bool_type, {2, 3}};

    auto input1 = mm->add_parameter("x1", sf);
    auto input2 = mm->add_parameter("x2", sb);
    auto cin1   = mm->add_instruction(
        migraphx::make_op("convert",
                            {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        input1);
    auto ret = mm->add_instruction(migraphx::make_op("equal"), cin1, input2);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("equal_bool_test.onnx");

    EXPECT(p == prog);
}
