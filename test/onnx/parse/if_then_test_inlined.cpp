
#include <onnx_test.hpp>

TEST_CASE(if_then_test_inlined)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    mm->add_literal(migraphx::literal(sc, {1}));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> ones(s.elements(), 1.0f);

    auto l1                 = mm->add_literal(s, ones);
    std::vector<float> rand = {-1.26487, -2.42279, 0.990835, 1.63072, 0.812238, -0.174946};

    mm->add_literal(s, rand);

    auto x = mm->add_parameter("x", s);
    mm->add_parameter("y", s);

    auto rt = mm->add_instruction(migraphx::make_op("add"), x, l1);
    mm->add_return({rt});

    auto prog = migraphx::parse_onnx("if_then_test_inlined.onnx");
    EXPECT(p == prog);
}
