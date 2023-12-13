
#include <onnx_test.hpp>


TEST_CASE(resize_nonstd_input_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<float> ds = {1.0f, 1.0f, 0.6f, 0.6f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal{ss, ds});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 4, 2}};
    auto inx = mm->add_parameter("X", sx);

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 1, 2}};
    std::vector<int> ind = {0, 4};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto tx =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inx);
    mm->add_instruction(migraphx::make_op("undefined"));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8}}}), tx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_nonstd_input_test.onnx");

    EXPECT(p == prog);
}


