
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>


TEST_CASE(averagepool_notset_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    auto ins   = mm->add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", migraphx::op::pooling_mode::average},
                                                        {"padding", {2, 2, 2, 2}},
                                                        {"stride", {2, 2}},
                                                        {"lengths", {6, 6}},
                                                        {"dilations", {1, 1}}}),
                                   input);
    auto ret   = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {2, 2}}}), ins);
    mm->add_return({ret});
    auto prog = migraphx::parse_onnx("averagepool_notset_test.onnx");

    EXPECT(p == prog);
}


