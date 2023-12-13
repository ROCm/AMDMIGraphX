
#include <onnx_test.hpp>


TEST_CASE(greaterorequal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input1 = mm->add_parameter("x1", migraphx::shape{migraphx::shape::float_type, {3}});
    auto input2 = mm->add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {3}});
    auto temp   = mm->add_instruction(migraphx::make_op("less"), input1, input2);
    auto bt     = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), temp);
    auto ge = mm->add_instruction(migraphx::make_op("not"), bt);

    mm->add_return({ge});

    auto prog = migraphx::parse_onnx("greaterorequal_test.onnx");
    EXPECT(p == prog);
}


