
#include <onnx_test.hpp>


TEST_CASE(castlike_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("0", migraphx::shape{migraphx::shape::half_type, {10}});
    mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {10}});
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l);

    auto prog = optimize_onnx("castlike_test.onnx");
    EXPECT(p == prog);
}


