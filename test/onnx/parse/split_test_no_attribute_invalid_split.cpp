
#include <onnx_test.hpp>


TEST_CASE(split_test_no_attribute_invalid_split)
{
    EXPECT(
        test::throws([&] { migraphx::parse_onnx("split_test_no_attribute_invalid_split.onnx"); }));
}


