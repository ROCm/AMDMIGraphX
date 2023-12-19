
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(clip_args_type_mismatch)
{
    auto p = migraphx::parse_onnx("clip_test_args_type_mismatch.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_0{migraphx::shape::float_type, {3, 3}};
    migraphx::parameter_map pp;
    std::vector<float> data_0 = {0.9, 1.2, 1.7, 1.9, 2.2, 2.7, 2.9, 3.2, 3.7};
    pp["0"]                   = migraphx::argument(s_0, data_0.data());
    auto result               = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.5, 2, 2, 1.9, 2.5, 3, 2.9, 3.2, 3.7};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


