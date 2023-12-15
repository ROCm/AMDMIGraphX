
#include <onnx_test.hpp>
#include <migraphx/op/pooling.hpp>

TEST_CASE(globalavgpool_dyn_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {3, 3}, {16, 16}, {16, 16}}});
    auto ret = mm->add_instruction(migraphx::make_op("pooling",
                                                     {{"mode", migraphx::op::pooling_mode::average},
                                                      {"lengths", {16, 16}},
                                                      {"padding", {0, 0, 0, 0}}}),
                                   input);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("globalavgpool_dyn_test.onnx", options);

    EXPECT(p == prog);
}
