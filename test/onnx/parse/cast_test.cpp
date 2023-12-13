
#include <onnx_test.hpp>


TEST_CASE(cast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, {10}});
    mm->add_instruction(
        migraphx::make_op("convert",
                          {{"target_type", migraphx::to_value(migraphx::shape::float_type)}}),
        l);

    auto prog = optimize_onnx("cast_test.onnx");
    EXPECT(p == prog);
}


