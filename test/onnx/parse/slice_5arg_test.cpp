
#include <onnx_test.hpp>


TEST_CASE(slice_5arg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {5, 5}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {1, 1}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, -2}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-1, -1}});
    mm->add_literal({{migraphx::shape::int32_type, {2}}, {-5, -3}});
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {-1, -2}}, {"starts", {-5, -3}}, {"ends", {-1, -1}}}),
        l0);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("slice_5arg_test.onnx");

    EXPECT(p == prog);
}


