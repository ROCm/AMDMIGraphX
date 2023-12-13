
#include <onnx_test.hpp>

TEST_CASE(logical_or_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::bool_type, {2, 3, 4, 5}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::bool_type, {2, 3, 4, 5}});
    auto ret = mm->add_instruction(migraphx::make_op("logical_or"), l0, l1);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("logical_or_test.onnx");

    EXPECT(p == prog);
}
