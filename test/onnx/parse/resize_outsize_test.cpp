
#include <onnx_test.hpp>

TEST_CASE(resize_outsize_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<int64_t> out_len = {1, 1, 4, 6};
    migraphx::shape so{migraphx::shape::int64_type, {4}};
    mm->add_literal(migraphx::literal(so, out_len));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto inx = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("undefined"));

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3};
    auto li              = mm->add_literal(migraphx::literal(si, ind));

    auto lrsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), inx);
    auto r    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lrsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("resize_outsize_test.onnx");

    EXPECT(p == prog);
}
