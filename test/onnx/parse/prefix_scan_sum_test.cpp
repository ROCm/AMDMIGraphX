
#include <onnx_test.hpp>


TEST_CASE(prefix_scan_sum)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {1}, {1}}, {0}});
    auto l0  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto ret = mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 0}, {"exclusive", true}, {"reverse", true}}),
        l0);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("prefix_scan_sum_test.onnx");
    EXPECT(p == prog);
}


