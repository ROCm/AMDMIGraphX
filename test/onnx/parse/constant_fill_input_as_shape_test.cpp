
#include <onnx_test.hpp>

TEST_CASE(constant_fill_input_as_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {2}}, {2, 3}});
    std::vector<std::size_t> dims(l0->get_shape().elements());
    migraphx::literal ls = l0->get_literal();
    ls.visit([&](auto s) { dims.assign(s.begin(), s.end()); });
    migraphx::shape s{migraphx::shape::float_type, dims};
    std::vector<float> value(s.elements(), 1.0);
    mm->add_literal(migraphx::literal{s, value});
    auto prog = optimize_onnx("constant_fill_input_as_shape_test.onnx");

    EXPECT(p == prog);
}
