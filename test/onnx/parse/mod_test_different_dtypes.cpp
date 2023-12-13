
#include <onnx_test.hpp>


TEST_CASE(mod_test_different_dtypes)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::int16_type, {3, 3, 3}});
    auto input1 = mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {3, 3, 3}});
    add_common_op(*mm, migraphx::make_op("mod"), {input0, input1});

    auto prog = optimize_onnx("mod_test_different_dtypes.onnx");

    EXPECT(p == prog);
}


