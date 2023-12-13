
#include <onnx_test.hpp>

TEST_CASE(group_norm_invalid_input_count_error_test)
{
    EXPECT(test::throws(
        [&] { migraphx::parse_onnx("group_norm_invalid_input_count_error_test.onnx"); }));
}
