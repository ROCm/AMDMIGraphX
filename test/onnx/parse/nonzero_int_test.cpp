
#include <onnx_test.hpp>

TEST_CASE(nonzero_int_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int16_type, {2, 3}};
    std::vector<int> data = {1, 1, 0, 1, 0, 1};
    mm->add_literal(migraphx::literal(s, data.begin(), data.end()));

    migraphx::shape si{migraphx::shape::int64_type, {2, 4}};
    std::vector<int64_t> indices = {0, 0, 1, 1, 0, 1, 0, 2};
    auto r                       = mm->add_literal(migraphx::literal(si, indices));
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("nonzero_int_test.onnx");
    EXPECT(p == prog);
}
