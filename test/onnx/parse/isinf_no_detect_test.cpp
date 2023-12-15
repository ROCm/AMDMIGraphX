
#include <onnx_test.hpp>


TEST_CASE(isinf_no_detect_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    mm->add_parameter("t1", s);
    auto ret = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}),
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::bool_type}, {false}}));
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("isinf_no_detect_test.onnx");
    EXPECT(p == prog);
}


