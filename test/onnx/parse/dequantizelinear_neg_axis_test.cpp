
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(dequantizelinear_neg_axis_test)
{
    migraphx::program p = make_dequantizelinear_axis_prog();

    auto prog = optimize_onnx("dequantizelinear_neg_axis_test.onnx", true);
    EXPECT(p.sort() == prog.sort());
}


