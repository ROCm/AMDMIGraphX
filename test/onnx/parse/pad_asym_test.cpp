
#include <onnx_test.hpp>


TEST_CASE(pad_asym_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 1, 0, 3, 0, 2, 0, 4}}}), l0);
    auto prog = optimize_onnx("pad_asym_test.onnx");

    EXPECT(p == prog);
}


