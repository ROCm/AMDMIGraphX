
#include <onnx_test.hpp>

TEST_CASE(neg_dynamic_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {{1, 10}, {3, 3}}};
    auto input = mm->add_parameter("0", s);
    auto ret   = mm->add_instruction(migraphx::make_op("neg"), input);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 10};
    auto prog                     = migraphx::parse_onnx("neg_dynamic_test.onnx", options);
    EXPECT(p == prog);
}
