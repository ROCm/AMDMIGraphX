
#include <onnx_test.hpp>


TEST_CASE(upsample_ver7_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto ix = mm->add_parameter("X", sx);

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};

    auto li  = mm->add_literal(migraphx::literal(si, ind));
    auto rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), ix);
    auto r   = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, li);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("upsample_ver7_test.onnx");

    EXPECT(p == prog);
}


