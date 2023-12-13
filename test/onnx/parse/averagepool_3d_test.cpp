
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(averagepool_3d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 5, 5, 5}});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"padding", {0, 0, 0, 0, 0, 0}},
                                           {"stride", {1, 1, 1}},
                                           {"lengths", {3, 3, 3}},
                                           {"dilations", {1, 1, 1}}}),
                        l0);

    auto prog = optimize_onnx("averagepool_3d_test.onnx");
    EXPECT(p == prog);
}
