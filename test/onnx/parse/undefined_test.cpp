
#include <onnx_test.hpp>


TEST_CASE(undefined_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}});
    auto l1 = mm->add_instruction(migraphx::make_op("undefined"));
    auto l2 = mm->add_instruction(migraphx::make_op("identity"), l1);
    mm->add_return({l2});

    auto prog = migraphx::parse_onnx("undefined_test.onnx");

    EXPECT(p == prog);
}


