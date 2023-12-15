
#include <onnx_test.hpp>


TEST_CASE(split_test_invalid_num_outputs)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("split_test_invalid_num_outputs.onnx"); }));
}


