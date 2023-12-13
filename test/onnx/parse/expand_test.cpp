
#include <onnx_test.hpp>

TEST_CASE(expand_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s(migraphx::shape::float_type, {3, 1, 1});
    auto param = mm->add_parameter("x", s);
    migraphx::shape ss(migraphx::shape::int32_type, {4});
    mm->add_literal(migraphx::literal(ss, {2, 3, 4, 5}));
    mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 5}}}), param);

    auto prog = optimize_onnx("expand_test.onnx");
    EXPECT(p == prog);
}
