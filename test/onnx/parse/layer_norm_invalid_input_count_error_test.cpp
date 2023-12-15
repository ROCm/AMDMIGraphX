
#include <onnx_test.hpp>


TEST_CASE(layer_norm_invalid_input_count_error_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("layer_norm_invalid_input_count_error_test.onnx"); }));
}


