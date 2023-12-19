
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(dynamicquantizelinear_1d_test)
{
    auto p = migraphx::parse_onnx("dynamicquantizelinear_1d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{0, 2, -3, -2.5, 1.34, 0.5};
    migraphx::shape s_x{migraphx::shape::float_type, {6}};
    migraphx::parameter_map pp;
    pp["x"]      = migraphx::argument(s_x, data.data());
    auto results = p.eval(pp);

    std::vector<uint8_t> y_results;
    results.at(0).visit([&](auto output) { y_results.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_gold = {153, 255, 0, 26, 221, 179};
    EXPECT(migraphx::verify::verify_rms_range(y_results, y_gold));

    std::vector<float> y_scale;
    results.at(1).visit([&](auto output) { y_scale.assign(output.begin(), output.end()); });
    std::vector<float> y_scale_gold = {0.0196078438};
    EXPECT(migraphx::verify::verify_rms_range(y_scale, y_scale_gold));

    std::vector<uint8_t> y_zpt;
    results.at(2).visit([&](auto output) { y_zpt.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_zpt_gold = {153};
    EXPECT(migraphx::verify::verify_rms_range(y_zpt, y_zpt_gold));
}

TEST_CASE(dynamicquantizelinear_1d_max_adjusted_test)
{
    auto p = migraphx::parse_onnx("dynamicquantizelinear_1d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{-1.0, -2.1, -1.3, -2.5, -3.34, -4.0};
    migraphx::shape s_x{migraphx::shape::float_type, {6}};
    migraphx::parameter_map pp;
    pp["x"]      = migraphx::argument(s_x, data.data());
    auto results = p.eval(pp);

    std::vector<uint8_t> y_results;
    results.at(0).visit([&](auto output) { y_results.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_gold = {191, 121, 172, 96, 42, 0};
    EXPECT(migraphx::verify::verify_rms_range(y_results, y_gold));

    std::vector<float> y_scale;
    results.at(1).visit([&](auto output) { y_scale.assign(output.begin(), output.end()); });
    std::vector<float> y_scale_gold = {0.0156862754};
    EXPECT(migraphx::verify::verify_rms_range(y_scale, y_scale_gold));

    std::vector<uint8_t> y_zpt;
    results.at(2).visit([&](auto output) { y_zpt.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_zpt_gold = {255};
    EXPECT(migraphx::verify::verify_rms_range(y_zpt, y_zpt_gold));
}


