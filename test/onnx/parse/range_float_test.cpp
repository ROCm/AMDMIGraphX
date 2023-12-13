
#include <onnx_test.hpp>

TEST_CASE(range_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(float{2});
    mm->add_literal(float{11});
    mm->add_literal(float{2});
    mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {5}}, {2, 4, 6, 8, 10}});

    auto prog = optimize_onnx("range_float_test.onnx");

    EXPECT(p == prog);
}
