
#include <onnx_test.hpp>


TEST_CASE(reducemean_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    auto l1  = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), l1);
    auto prog = optimize_onnx("reducemean_test.onnx");

    EXPECT(p == prog);
}


