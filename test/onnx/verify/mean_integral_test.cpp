
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(mean_integral_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_integral_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 2}};
    const int num_elms = 8;
    const int num_data = 10;
    const std::vector<int> scalars{1, 5, 14, 2, 6, 21, 101, 0, -4, -11};
    std::vector<std::vector<int>> data;
    std::transform(scalars.begin(), scalars.end(), std::back_inserter(data), [&](const auto i) {
        return std::vector<int>(num_elms, i);
    });

    migraphx::parameter_map pp;
    for(std::size_t i = 0; i < num_data; ++i)
        pp[std::to_string(i)] = migraphx::argument(s, data[i].data());

    auto result = p.eval(pp).back();
    std::vector<double> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    const auto mean = std::accumulate(scalars.begin(), scalars.end(), 0) / num_data;
    std::vector<int> gold(num_elms, mean);
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


