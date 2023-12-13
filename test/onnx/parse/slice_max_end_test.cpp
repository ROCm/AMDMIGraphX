
#include <onnx_test.hpp>

TEST_CASE(slice_max_end_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {10, 20}});
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1}}, {"starts", {1, 2}}, {"ends", {3000000000, -1}}}),
        l0);
    auto prog = optimize_onnx("slice_max_end_test.onnx");

    EXPECT(p == prog);
}
