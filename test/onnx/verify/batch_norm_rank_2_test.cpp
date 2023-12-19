
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(batch_norm_rank_2_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_rank_2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 5}};
    migraphx::shape c_shape(migraphx::shape::float_type, {5});
    std::vector<float> x_data = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> scale_data(5, 1.);
    std::vector<float> bias_data(5, 0.);
    std::vector<float> mean_data = {1., 2., 1., 2., 1.};
    std::vector<float> variance_data(5, 0.5);

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.,
                               0.,
                               2.8284243,
                               2.8284243,
                               5.65684859,
                               7.07106074,
                               7.07106074,
                               9.89948504,
                               9.89948504,
                               12.72790933};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


