
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(lppool_l2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 3, 5}});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::lpnorm},
                                           {"padding", {0, 0}},
                                           {"stride", {1}},
                                           {"lengths", {3}},
                                           {"dilations", {1}},
                                           {"lp_order", 2}}),
                        l0);
    auto prog = optimize_onnx("lppool_l2_test.onnx");
    EXPECT(p == prog);
}
