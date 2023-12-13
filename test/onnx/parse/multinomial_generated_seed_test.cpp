
#include <onnx_test.hpp>

TEST_CASE(multinomial_generated_seed_test)
{
    // multinomial op. no longer generates its own randoms
    auto p1 = optimize_onnx("multinomial_generated_seed_test.onnx");
    auto p2 = optimize_onnx("multinomial_generated_seed_test.onnx");

    EXPECT(p1 == p2);
}
