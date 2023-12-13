
#include <onnx_test.hpp>

TEST_CASE(clip_test_op11_no_args)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_onnx("clip_test_op11_no_args.onnx");

    EXPECT(p == prog);
}
