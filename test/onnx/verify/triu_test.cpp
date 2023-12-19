
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>


TEST_CASE(triu_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


