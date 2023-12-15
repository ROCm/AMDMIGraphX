
#include <onnx_test.hpp>


TEST_CASE(shape_start_oob_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4, {1, 4}}, {4, 4}, {2, 4}, {2, 4}}};
    auto p0 = mm->add_parameter("x", s);
    migraphx::shape s_shape{migraphx::shape::int64_type, {4}};
    auto ret = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 4}}), p0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{1, 4, {1, 4}}, {4, 4}, {2, 4}, {2, 4}};
    auto prog                       = migraphx::parse_onnx("shape_start_oob_test.onnx", options);

    EXPECT(p == prog);
}


