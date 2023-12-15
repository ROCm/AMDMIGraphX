
#include <onnx_test.hpp>

TEST_CASE(gather_scalar_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0 = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    std::vector<size_t> idims{1};
    auto l1 =
        mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, idims, {0}});
    int axis = 1;
    mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), l0, l1);
    auto prog = optimize_onnx("gather_scalar_test.onnx");

    EXPECT(p == prog);
}
