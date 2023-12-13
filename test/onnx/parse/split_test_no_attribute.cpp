
#include <onnx_test.hpp>

TEST_CASE(split_test_no_attribute)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape si{migraphx::shape::int64_type, {4}, {1}};
    std::vector<int> ind = {75, 75, 75, 75};

    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {300, 15}});
    mm->add_literal(migraphx::literal(si, ind));
    auto r1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {75}}}), input);
    auto r2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {75}}, {"ends", {150}}}), input);
    auto r3 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {150}}, {"ends", {225}}}), input);
    auto r4 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {225}}, {"ends", {300}}}), input);

    mm->add_return({r1, r2, r3, r4});

    auto prog = migraphx::parse_onnx("split_test_no_attribute.onnx");
    EXPECT(p == prog);
}
