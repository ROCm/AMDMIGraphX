
#include <onnx_test.hpp>


TEST_CASE(constant_multiple_attributes_test)
{
    EXPECT(test::throws([&] { optimize_onnx("constant_multiple_attributes_test.onnx"); }));
}


