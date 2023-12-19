
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(triu_batch_diff_k_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_batch_diff_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {2, 2, 3}}, p);

    std::vector<float> gold = {0, 0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


