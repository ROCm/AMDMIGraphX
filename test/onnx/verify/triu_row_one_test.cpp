
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(triu_row_one_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_row_one_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {1, 4}}, p);

    std::vector<float> gold = {0, 2, 3, 4};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
