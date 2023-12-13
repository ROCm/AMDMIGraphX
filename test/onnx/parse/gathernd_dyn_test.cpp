
#include <onnx_test.hpp>

TEST_CASE(gathernd_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {{2, 4, {2}}, {2, 4}}});
    auto l1 = mm->add_parameter("indices",
                                migraphx::shape{migraphx::shape::int64_type, {{1, 3}, {2, 2}}});
    auto r  = mm->add_instruction(migraphx::make_op("gathernd"), l0, l1);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"]    = {{2, 4, {2}}, {2, 4}};
    options.map_dyn_input_dims["indices"] = {{1, 3}, {2, 2}};
    auto prog                             = migraphx::parse_onnx("gathernd_dyn_test.onnx", options);
    EXPECT(p == prog);
}
