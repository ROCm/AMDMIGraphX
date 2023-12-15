
#include <onnx_test.hpp>


TEST_CASE(reversesequence_sequence_lens_shape_err_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("reversesequence_sequence_lens_shape_err_test.onnx"); }));
}


