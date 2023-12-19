
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(resize_upsample_linear_ac_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_ac_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1,
                               4.0f / 3,
                               5.0f / 3,
                               2,
                               5.0f / 3,
                               2,
                               7.0f / 3,
                               8.0f / 3,
                               7.0f / 3,
                               8.0f / 3,
                               3,
                               10.0f / 3,
                               3,
                               10.0f / 3,
                               11.0f / 3,
                               4};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
