
#include <onnx_test.hpp>


TEST_CASE(if_then_else_multi_output_shapes_inlined_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    mm->add_literal(migraphx::literal(sc, {1}));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s_trail{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> ones(s.elements(), 1.0f);

    auto l1                 = mm->add_literal(s_trail, ones);
    std::vector<float> rand = {-1.01837, -0.305541, -0.254105, 0.892955, 1.38714, -0.584205};
    mm->add_literal(s, rand);

    auto x = mm->add_parameter("x", s_trail);
    mm->add_parameter("y", s);

    auto rt  = mm->add_instruction(migraphx::make_op("add"), x, l1);
    auto rt2 = mm->add_instruction(migraphx::make_op("add"), x, x);

    mm->add_return({rt, rt2});

    auto prog = migraphx::parse_onnx("if_then_else_multi_output_shapes_inlined_test.onnx");
    EXPECT(p == prog);
}


