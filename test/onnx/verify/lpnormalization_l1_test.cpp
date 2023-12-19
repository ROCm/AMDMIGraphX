
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(lpnormalization_1norm)
{
    migraphx::program p = migraphx::parse_onnx("lpnormalization_l1_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{0.f, 2.f, -2.f, 1.f, 1.f, -5.f, 3.f, -1.f, -4.f, 3.f, 0.f, 0.f};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{0.f,
                            2.f / 5.f,
                            -2.f / 5.f,
                            1.f / 5.f,
                            1.f / 10.f,
                            -5.f / 10.f,
                            3.f / 10.f,
                            -1.f / 10.f,
                            -4.f / 7.f,
                            3.f / 7.f,
                            0.f,
                            0.f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
