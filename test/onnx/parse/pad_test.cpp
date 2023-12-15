
#include <onnx_test.hpp>


TEST_CASE(pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), l0);
    auto prog = optimize_onnx("pad_test.onnx");

    EXPECT(p == prog);
}


