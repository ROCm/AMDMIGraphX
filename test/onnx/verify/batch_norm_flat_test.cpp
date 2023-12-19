
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(batch_norm_flat_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_flat_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {10}};
    migraphx::shape c_shape(migraphx::shape::float_type, {1});
    std::vector<float> x_data        = {1.6524342,
                                        -0.51048076,
                                        0.32543048,
                                        2.4410043,
                                        2.0833702,
                                        0.44981122,
                                        1.0044622,
                                        -0.24006313,
                                        -0.43065986,
                                        0.07626268};
    std::vector<float> scale_data    = {-0.02927135};
    std::vector<float> bias_data     = {0.42347777};
    std::vector<float> mean_data     = {-0.00449735};
    std::vector<float> variance_data = {0.5184545};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.35612,
                               0.44404706,
                               0.4100655,
                               0.32406294,
                               0.33860153,
                               0.40500915,
                               0.38246143,
                               0.43305403,
                               0.4408022,
                               0.42019472};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


