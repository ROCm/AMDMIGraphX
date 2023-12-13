
#include <onnx_test.hpp>


TEST_CASE(const_of_shape_dyn_float_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto od_param =
        mm->add_parameter("output_dims", migraphx::shape{migraphx::shape::int64_type, {3}});
    auto alloc_ins = mm->add_instruction(
        migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}), od_param);
    migraphx::shape dv_shape(migraphx::shape::float_type, {1}, {0});
    auto dv_lit   = mm->add_literal(migraphx::literal(dv_shape, {10}));
    auto fill_ins = mm->add_instruction(migraphx::make_op("fill"), dv_lit, alloc_ins);
    mm->add_return({fill_ins});

    migraphx::onnx_options options;
    auto prog = parse_onnx("const_of_shape_dyn_float_test.onnx", options);
    EXPECT(p == prog);
}


