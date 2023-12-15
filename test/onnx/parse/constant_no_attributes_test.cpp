
#include <onnx_test.hpp>

TEST_CASE(constant_no_attributes_test)
{
    EXPECT(test::throws([&] { optimize_onnx("constant_no_attributes_test.onnx"); }));
}
