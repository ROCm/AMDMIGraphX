
#include <onnx_test.hpp>

TEST_CASE(isnan_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 3}};
    auto t1  = mm->add_parameter("t1", s);
    auto ret = mm->add_instruction(migraphx::make_op("isnan"), t1);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("isnan_half_test.onnx");
    EXPECT(p == prog);
}
