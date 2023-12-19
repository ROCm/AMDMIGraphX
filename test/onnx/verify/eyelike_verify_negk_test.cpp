
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(eyelike_verify_negk_test)
{
    migraphx::program p = migraphx::parse_onnx("eyelike_verify_negk_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{12, 0};
    migraphx::parameter_map pp;
    pp["T1"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_eyelike_mat = {0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold_eyelike_mat));
}


