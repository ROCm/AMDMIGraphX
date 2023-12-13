
#include <onnx_test.hpp>


TEST_CASE(where_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lc  = mm->add_parameter("c", migraphx::shape{migraphx::shape::bool_type, {2}});
    auto lx  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto ly  = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 1, 2, 2}});

    auto lccm =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2}}}), lc);
    auto lxm =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2}}}), lx);
    auto lym =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2}}}), ly);

    auto r = mm->add_instruction(migraphx::make_op("where"), lccm, lxm, lym);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("where_test.onnx");

    EXPECT(p == prog);
}


