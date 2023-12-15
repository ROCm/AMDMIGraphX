
#include <onnx_test.hpp>


TEST_CASE(reversesequence_time_axis_err_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("reversesequence_time_axis_err_test.onnx"); }));
}


