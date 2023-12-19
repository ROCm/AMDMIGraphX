
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(multinomial_dyn_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = migraphx::parse_onnx("multinomial_dyn_test.onnx", options);
    const size_t batch_size(2);
    const size_t categories(5);
    const size_t sample_size(100000);
    p.compile(migraphx::make_target("ref"));

    // Distribution function (2 distributions of 5 categories each)
    std::vector<int> dist{15, 25, 15, 25, 20, 20, 20, 10, 25, 25};
    EXPECT(dist.size() == categories * batch_size);
    std::vector<float> data(categories * batch_size);

    std::transform(dist.begin(), dist.end(), data.begin(), [&](auto d) { return log(d); });
    // Shape of the probability distribution, which also defines the number of categories
    migraphx::shape s{migraphx::shape::float_type, {batch_size, categories}};

    migraphx::parameter_map pp;
    pp["input"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();

    std::vector<int32_t> result_vec(batch_size * sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Make a categorical histogram of output
    // for first result in batch
    std::vector<int> res_dist(categories, 0);
    size_t r = 0;
    for(r = 0; r < result_vec.size() / 2; r++)
        res_dist[result_vec[r]]++;

    // normalizing factors for original and measured distributions
    auto dist_sum     = std::accumulate(dist.begin(), dist.begin() + 5, 0);
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);

    //  Values approximate the distribution in dist
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);

    std::transform(dist.begin(), dist.begin() + 5, norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        norm, migraphx::verify::expected{res_norm}, migraphx::verify::tolerance{0.01}));

    // Make a categorical histogram of output
    // for second result in batch
    std::fill(res_dist.begin(), res_dist.end(), 0);
    for(; r < result_vec.size(); r++)
        res_dist[result_vec[r]]++;

    dist_sum     = std::accumulate(dist.begin() + 5, dist.end(), 0);
    res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::transform(dist.begin() + 5, dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));
}
