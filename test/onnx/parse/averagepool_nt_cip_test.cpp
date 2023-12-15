
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(averagepool_nt_cip_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 5}});
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0, 1, 1};
    auto ins_pad = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), input);
    auto ret     = mm->add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", migraphx::op::pooling_mode::average},
                                                          {"padding", {0, 0, 0, 0}},
                                                          {"stride", {2, 2}},
                                                          {"lengths", {6, 6}},
                                                          {"dilations", {1, 1}}}),
                                   ins_pad);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    EXPECT(p == prog);
}
