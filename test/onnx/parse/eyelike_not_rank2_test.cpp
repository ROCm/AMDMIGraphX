
#include <onnx_test.hpp>

TEST_CASE(eyelike_not_rank2_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("eyelike_not_rank2_test.onnx"); }));
}
