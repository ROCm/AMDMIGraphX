
#include <onnx_test.hpp>
#include <migraphx/op/common.hpp>

TEST_CASE(averagepool_same_lower_test)
{
    // auto_pad mode of SAME_LOWER with a static input shape is handled in parsing and
    // padding_mode is set to default_ when the operation is created
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = mm->add_instruction(
        migraphx::make_op("pooling",
                            {
                              {"mode", migraphx::op::pooling_mode::average},
                              {"padding", {1, 1, 1, 1}},
                              {"stride", {1, 1}},
                              {"lengths", {2, 2}},
                              {"dilations", {1, 1}},
                              {"padding_mode", migraphx::op::padding_mode_t::default_},
                          }),
        input);
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {0, 0}}, {"ends", {5, 5}}}), ins);
    mm->add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_same_lower_test.onnx");

    EXPECT(p == prog);
}
