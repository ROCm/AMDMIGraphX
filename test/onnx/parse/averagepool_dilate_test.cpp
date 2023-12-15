
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(averagepool_dilate_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 4, 3}});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::average},
                                           {"padding", {1, 1}},
                                           {"stride", {1}},
                                           {"lengths", {2}},
                                           {"dilations", {3}}}),
                        input);

    auto prog = optimize_onnx("averagepool_dilate_test.onnx");

    EXPECT(p == prog);
}
