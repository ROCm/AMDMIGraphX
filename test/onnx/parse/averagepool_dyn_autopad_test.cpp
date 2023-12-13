
#include <onnx_test.hpp>
#include <migraphx/op/common.hpp>


TEST_CASE(averagepool_dyn_autopad_test)
{
    // Pooling with dynamic input and auto padding. Default padding values will be overridden.
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", {migraphx::shape::float_type, {{1, 4}, {3, 3}, {5, 5}, {5, 5}, {5, 5}}});
    auto ret = mm->add_instruction(
        migraphx::make_op("pooling",
                          {
                              {"mode", migraphx::op::pooling_mode::average},
                              {"stride", {2, 2, 2}},
                              {"lengths", {3, 3, 3}},
                              {"dilations", {1, 1, 1}},
                              {"padding", {0, 0, 0, 0, 0, 0}},
                              {"padding_mode", migraphx::op::padding_mode_t::same_upper},
                          }),
        l0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog = migraphx::parse_onnx("averagepool_dyn_autopad_test.onnx", options);
    EXPECT(p == prog);
}


