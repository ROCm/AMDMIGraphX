
#ifndef MIGRAPHX_GUARD_TEST_ONNX_ONNX_VERIFY_UTILS_HPP
#define MIGRAPHX_GUARD_TEST_ONNX_ONNX_VERIFY_UTILS_HPP

#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

template <typename T = float>
std::vector<T> norm_test(const std::vector<size_t>& x_dims,
                         std::vector<T>& scale,
                         std::vector<T>& bias,
                         migraphx::program p)
{
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::get_type<T>{}, x_dims};
    migraphx::shape s_s{migraphx::shape::get_type<T>{}, {scale.size()}};
    migraphx::shape s_b{migraphx::shape::get_type<T>{}, {scale.size()}};

    std::vector<T> x(s_x.elements());
    std::iota(std::begin(x), std::end(x), 1);

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["scale"] = migraphx::argument(s_s, scale.data());
    pp["bias"]  = migraphx::argument(s_b, bias.data());

    auto result = p.eval(pp).back();

    std::vector<T> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}

template <typename T = float>
std::vector<T> mvn_test(std::vector<size_t> data_lens, migraphx::program p)
{
    p.compile(migraphx::make_target("ref"));

    migraphx::shape data_shape(migraphx::shape::get_type<T>{}, std::move(data_lens));
    std::vector<T> data(data_shape.elements());
    std::iota(begin(data), end(data), 0);

    migraphx::parameter_map pm;
    pm["data"] = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pm).back();
    std::vector<T> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}

inline std::vector<float> gen_trilu_test(const migraphx::shape& s, const migraphx::program& p)
{
    // input data filled with values 1 to nelements
    std::vector<float> x_data(s.elements());
    std::iota(x_data.begin(), x_data.end(), 1);

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}

#endif
