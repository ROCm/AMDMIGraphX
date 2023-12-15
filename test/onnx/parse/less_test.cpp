
#include <onnx_test.hpp>

TEST_CASE(less_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto input1 = mm->add_literal(migraphx::literal(s, data));
    auto input2 = mm->add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto le     = mm->add_instruction(migraphx::make_op("less"), input1, input2);
    auto ret    = mm->add_instruction(
        migraphx::make_op("convert",
                             {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        le);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("less_test.onnx");
    EXPECT(p == prog);
}
