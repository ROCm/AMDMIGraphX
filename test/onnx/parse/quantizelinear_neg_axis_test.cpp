
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>


TEST_CASE(quantizelinear_neg_axis_test)
{
    migraphx::program p = make_quantizelinear_axis_prog();

    auto prog = optimize_onnx("quantizelinear_neg_axis_test.onnx", true);
    EXPECT(p.sort() == prog.sort());
}


