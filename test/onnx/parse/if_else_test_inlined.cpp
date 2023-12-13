
#include <onnx_test.hpp>

TEST_CASE(if_else_test_inlined)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    mm->add_literal(migraphx::literal(sc, {0}));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> ones(s.elements(), 1.0f);
    mm->add_literal(s, ones);

    std::vector<float> rand = {0.811412, -0.949771, -0.169276, 0.36552, -0.14801, 2.07061};
    auto l2                 = mm->add_literal(s, rand);

    mm->add_parameter("x", s);

    auto y  = mm->add_parameter("y", s);
    auto re = mm->add_instruction(migraphx::make_op("mul"), y, l2);
    mm->add_return({re});

    auto prog = migraphx::parse_onnx("if_else_test_inlined.onnx");
    EXPECT(p == prog);
}
