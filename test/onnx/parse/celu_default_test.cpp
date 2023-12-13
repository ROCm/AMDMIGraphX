
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(celu_default_test)
{
    migraphx::program p;
    auto* mm                            = p.get_main_module();
    std::vector<std::size_t> input_lens = {2, 3};
    auto input_type                     = migraphx::shape::float_type;
    migraphx::shape s{input_type, input_lens};
    float alpha = 1.0;
    add_celu_instruction(mm, s, alpha);
    auto prog = optimize_onnx("celu_default_test.onnx");
    EXPECT(p == prog);
}
