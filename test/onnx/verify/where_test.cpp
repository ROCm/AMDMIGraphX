
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(where_test)
{
    migraphx::program p = migraphx::parse_onnx("where_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape c_shape{migraphx::shape::bool_type, {2}};
    std::vector<int8_t> c_data = {1, 0};

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> x_data(8, 1.0f);

    migraphx::shape y_shape{migraphx::shape::float_type, {2, 1, 2, 2}};
    std::vector<float> y_data(8, 2.0f);

    migraphx::parameter_map pp;
    pp["c"] = migraphx::argument(c_shape, c_data.data());
    pp["x"] = migraphx::argument(x_shape, x_data.data());
    pp["y"] = migraphx::argument(y_shape, y_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


