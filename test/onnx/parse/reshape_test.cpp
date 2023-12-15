
#include <onnx_test.hpp>
#include <migraphx/op/reshape.hpp>

TEST_CASE(reshape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::op::reshape op;
    std::vector<int64_t> reshape_dims{3, 8};
    mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}}, reshape_dims});
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 2, 3}});
    op.dims = reshape_dims;
    mm->add_instruction(op, l0);
    mm->add_instruction(op, l0);
    auto prog = optimize_onnx("reshape_test.onnx");
    EXPECT(p == prog);
}
