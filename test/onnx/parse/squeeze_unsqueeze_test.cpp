
#include <onnx_test.hpp>

TEST_CASE(squeeze_unsqueeze_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int64_t> squeeze_axes{0, 2, 3, 5};
    std::vector<int64_t> unsqueeze_axes{0, 1, 3, 5};
    auto l0 =
        mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 1, 1, 2, 1}});
    auto l1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", squeeze_axes}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}), l1);
    auto prog = optimize_onnx("squeeze_unsqueeze_test.onnx");

    EXPECT(p == prog);
}
