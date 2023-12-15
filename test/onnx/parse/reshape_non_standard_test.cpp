
#include <onnx_test.hpp>
#include <migraphx/op/reshape.hpp>

TEST_CASE(reshape_non_standard_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::op::reshape op;
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    auto x = mm->add_parameter("x", s);
    auto tran_x =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 3, 2}}}), tran_x);
    auto prog = optimize_onnx("reshape_non_standard_test.onnx");

    EXPECT(p == prog);
}
