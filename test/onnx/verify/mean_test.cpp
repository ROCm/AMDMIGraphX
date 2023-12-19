
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(mean_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::double_type, {2, 2, 2}};
    const int num_elms = 8;
    const int num_data = 10;
    const std::vector<double> scalars{1.0, 2.0, -2.5, 3.3, 10.7, -1.0, 100.0, 7.9, 0.01, -56.8};
    std::vector<std::vector<double>> data;
    std::transform(scalars.begin(), scalars.end(), std::back_inserter(data), [&](const auto& i) {
        return std::vector<double>(num_elms, i);
    });

    migraphx::parameter_map pp;
    for(std::size_t i = 0; i < num_data; ++i)
        pp[std::to_string(i)] = migraphx::argument(s, data[i].data());

    auto result = p.eval(pp).back();
    std::vector<double> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    const auto mean = std::accumulate(scalars.begin(), scalars.end(), 0.0) / num_data;
    std::vector<double> gold(num_elms, mean);
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


