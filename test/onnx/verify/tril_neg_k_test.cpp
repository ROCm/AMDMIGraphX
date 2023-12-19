
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(tril_neg_k_test)
{
    migraphx::program p = migraphx::parse_onnx("tril_neg_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {0, 0, 0, 0, 5, 0, 0, 0, 9, 10, 0, 0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
