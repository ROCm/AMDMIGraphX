
#include <onnx_test.hpp>


TEST_CASE(embedding_bag_offset_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("embedding_bag_offset_test.onnx"); }));
}


