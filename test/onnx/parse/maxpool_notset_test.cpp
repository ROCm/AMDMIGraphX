
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(maxpool_notset_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    mm->add_instruction(migraphx::make_op("pooling",
                                          {{"mode", migraphx::op::pooling_mode::max},
                                           {"padding", {0, 0, 1, 1}},
                                           {"stride", {2, 2}},
                                           {"lengths", {6, 6}},
                                           {"dilations", {1, 1}}}),
                        input);

    auto prog = optimize_onnx("maxpool_notset_test.onnx");

    EXPECT(p == prog);
}
