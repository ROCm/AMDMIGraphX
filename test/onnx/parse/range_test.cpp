
#include <onnx_test.hpp>

TEST_CASE(range_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->add_literal(int64_t{10});
    mm->add_literal(int64_t{6});
    mm->add_literal(int64_t{-3});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {2}}, {10, 7}});

    auto prog = optimize_onnx("range_test.onnx");

    EXPECT(p == prog);
}
