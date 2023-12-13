
#include <onnx_test.hpp>


TEST_CASE(randomuniform_generated_seed_test)
{
    auto p1 = optimize_onnx("randomuniform_generated_seed_test.onnx");
    auto p2 = optimize_onnx("randomuniform_generated_seed_test.onnx");

    EXPECT(p1 != p2);
}


