
#include <onnx_test.hpp>


TEST_CASE(transpose_dyn_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}, {3, 3}}});
    std::vector<int64_t> perm{0, 3, 1, 2};
    auto t0 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), input);
    mm->add_return({t0});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto prog                     = migraphx::parse_onnx("transpose_dyn_test.onnx", options);

    EXPECT(p == prog);
}


