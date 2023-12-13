
#include <onnx_test.hpp>


TEST_CASE(pad_cnst_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{2, 4, {2}}, {2, 4, {2}}}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 0, 1}});
    auto ret = mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 2, 0, 1}}}), x);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 4, {2}}, {2, 4, {2}}};
    auto prog                       = parse_onnx("pad_cnst_dyn_test.onnx", options);
    EXPECT(p == prog);
}


