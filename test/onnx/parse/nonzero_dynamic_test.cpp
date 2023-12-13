
#include <onnx_test.hpp>

TEST_CASE(nonzero_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {2, 2}};
    auto data = mm->add_parameter("data", s);
    auto r    = mm->add_instruction(migraphx::make_op("nonzero"), data);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("nonzero_dynamic_test.onnx");
    EXPECT(p == prog);
}
