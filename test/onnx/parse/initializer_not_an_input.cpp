
#include <onnx_test.hpp>
#include <migraphx/apply_alpha_beta.hpp>


TEST_CASE(initializer_not_an_input)
{
    migraphx::program p;
    auto* mm             = p.get_main_module();
    std::vector<float> w = {1, 2, 3, 4, 5, 6, 7, 8};
    auto l1 = mm->add_literal(migraphx::literal({migraphx::shape::float_type, {2, 4}}, w));
    auto l0 = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {5, 2}});
    migraphx::add_apply_alpha_beta(*mm, {l0, l1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto prog = optimize_onnx("initializer_not_an_input.onnx");

    EXPECT(p == prog);
}


