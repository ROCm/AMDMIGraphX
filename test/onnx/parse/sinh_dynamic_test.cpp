
#include <onnx_test.hpp>

TEST_CASE(sinh_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{1, 10};
    std::vector<migraphx::shape::dynamic_dimension> dyn_dims;
    dyn_dims.push_back(dd);
    auto input = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dyn_dims});
    auto ret   = mm->add_instruction(migraphx::make_op("sinh"), input);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = dd;
    auto prog                     = parse_onnx("sinh_dynamic_test.onnx", options);

    EXPECT(p == prog);
}
