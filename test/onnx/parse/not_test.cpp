
#include <onnx_test.hpp>


TEST_CASE(not_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::int32_type, {4}});
    auto ret = mm->add_instruction(migraphx::make_op("not"), l0);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("not_test.onnx");

    EXPECT(p == prog);
}


