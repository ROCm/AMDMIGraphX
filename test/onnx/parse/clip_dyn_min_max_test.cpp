
#include <onnx_test.hpp>


TEST_CASE(clip_dyn_min_max_test)
{
    migraphx::program p;
    auto* mm                                            = p.get_main_module();
    auto min_val                                        = mm->add_literal(0.0f);
    auto max_val                                        = mm->add_literal(6.0f);
    std::vector<migraphx::shape::dynamic_dimension> dds = {{2, 8, {3}}};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, dds});
    min_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(dds)}}), min_val, l0);
    max_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(dds)}}), max_val, l0);
    auto ret = mm->add_instruction(migraphx::make_op("clip"), l0, min_val, max_val);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 8, {3}};
    auto prog                     = parse_onnx("clip_dyn_min_max_test.onnx", options);

    EXPECT(p == prog);
}


