
#include <onnx_test.hpp>

TEST_CASE(neg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {2, 3}};
    auto input = mm->add_parameter("0", s);
    auto ret   = mm->add_instruction(migraphx::make_op("neg"), input);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("neg_test.onnx");

    EXPECT(p == prog);
}
