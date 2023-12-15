
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(globallppool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraphx::op::pooling{migraphx::op::pooling_mode::lpnorm};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    op.padding = {0, 0, 0, 0};
    mm->add_instruction(op, input);

    auto prog = optimize_onnx("globallppool_test.onnx");

    EXPECT(p == prog);
}
