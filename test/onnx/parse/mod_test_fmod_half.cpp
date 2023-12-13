
#include <onnx_test.hpp>


TEST_CASE(mod_test_fmod_half)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::half_type, {3, 3, 3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::half_type, {3, 3, 3}});
    mm->add_instruction(migraphx::make_op("fmod"), input0, input1);

    auto prog = optimize_onnx("mod_test_fmod_half.onnx");

    EXPECT(p == prog);
}


