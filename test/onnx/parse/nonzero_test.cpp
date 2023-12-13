
#include <onnx_test.hpp>

TEST_CASE(nonzero_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    std::vector<float> data = {1, 0, 1, 1};
    mm->add_literal(migraphx::literal(s, data));

    migraphx::shape si{migraphx::shape::int64_type, {2, 3}};
    std::vector<int64_t> indices = {0, 1, 1, 0, 0, 1};
    auto r                       = mm->add_literal(migraphx::literal(si, indices));
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("nonzero_test.onnx");
    EXPECT(p == prog);
}
