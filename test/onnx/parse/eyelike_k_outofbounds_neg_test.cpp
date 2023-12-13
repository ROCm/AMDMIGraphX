
#include <onnx_test.hpp>

TEST_CASE(eyelike_k_outofbounds_neg_test)
{
    EXPECT(test::throws([&] { migraphx::parse_onnx("eyelike_k_outofbounds_neg_test.onnx"); }));
}
