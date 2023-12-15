
#include <onnx_test.hpp>


TEST_CASE(squeeze_unsqueeze_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int64_t> squeeze_axes{0, 2, 3, 5};
    std::vector<int64_t> unsqueeze_axes{0, 1, 3, 5};
    auto l0  = mm->add_parameter("0",
                                migraphx::shape{migraphx::shape::float_type,
                                                 {{1, 1}, {1, 4}, {1, 1}, {1, 1}, {1, 4}, {1, 1}}});
    auto c0  = mm->add_instruction(migraphx::make_op("contiguous"), l0);
    auto l1  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", squeeze_axes}}), c0);
    auto c1  = mm->add_instruction(migraphx::make_op("contiguous"), l1);
    auto ret = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", unsqueeze_axes}}), c1);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = parse_onnx("squeeze_unsqueeze_dyn_test.onnx", options);

    EXPECT(p == prog);
}


