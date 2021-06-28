#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

TEST_CASE(averagepool_notset_test)
{
    auto p = migraphx::parse_onnx("averagepool_notset_test.onnx");
    p.compile(migraphx::ref::target{});
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {12};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(averagepool_nt_cip_test)
{
    auto p = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    p.compile(migraphx::ref::target{});
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {8.33333};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(gather_elements)
{
    migraphx::program p = migraphx::parse_onnx("gather_elements_axis0_test.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s_data{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data = {
        0.25, 0.75, 0.9375, 0.4375, 0.6875, 0.5625, -0.875, 0.1875, -0.125, 0.5, -0.9375, -0.0625};

    migraphx::shape s_ind{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> ind = {2, 1, 2, 0, 1, 0};

    migraphx::parameter_map pp;
    pp["data"]    = migraphx::argument(s_data, data.data());
    pp["indices"] = migraphx::argument(s_ind, ind.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-0.125, 0.5625, -0.9375, 0.25, 0.5625, 0.9375};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(if_else_test)
{
    migraphx::program p = migraphx::parse_onnx("if_else_test.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_data, data.data());
    pp["y"] = migraphx::argument(s_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -0.0364609435, 0.475317657, -0.00417715637, -0.0599277429, 0.0755792186, -0.0218581557};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(if_literal_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_literal_test.onnx");
        p.compile(migraphx::ref::target{});
        migraphx::shape s_data{migraphx::shape::bool_type};
        std::vector<char> data = {static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["cond"] = migraphx::argument(s_data, data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

        return result_vector;
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {1, 2, 3, 4, 5};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {5, 4, 3, 2, 1};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }
}

TEST_CASE(if_pl_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_pl_test.onnx");
        p.compile(migraphx::ref::target{});
        migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
        migraphx::shape ys{migraphx::shape::float_type, {3, 3}};
        migraphx::shape cond_s{migraphx::shape::bool_type};

        std::vector<float> x_data(xs.elements(), 1.0f);
        std::vector<float> y_data(ys.elements(), 2.0f);
        std::vector<char> cond_data{static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["x"]    = migraphx::argument(xs, x_data.data());
        pp["y"]    = migraphx::argument(ys, y_data.data());
        pp["cond"] = migraphx::argument(cond_s, cond_data.data());

        auto result = p.eval(pp).back();
        std::vector<float> ret;
        result.visit([&](auto output) { ret.assign(output.begin(), output.end()); });

        return ret;
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {2, 3, 4, 5, 6, 7};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {1, 2, 3, 4, 5, 6};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }
}

TEST_CASE(if_tuple_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_tuple_test.onnx");
        p.compile(migraphx::ref::target{});
        migraphx::shape xs{migraphx::shape::float_type, {1, 4}};
        migraphx::shape ys{migraphx::shape::float_type, {3, 4}};
        migraphx::shape cond_s{migraphx::shape::bool_type};

        std::vector<float> x_data(xs.elements(), 1.0f);
        std::vector<float> y_data(ys.elements(), 2.0f);
        std::vector<char> cond_data{static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["x"]    = migraphx::argument(xs, x_data.data());
        pp["y"]    = migraphx::argument(ys, y_data.data());
        pp["cond"] = migraphx::argument(cond_s, cond_data.data());

        auto results = p.eval(pp);
        std::vector<std::vector<float>> rets;
        for(const auto& arg : results)
        {
            std::vector<float> vec;
            arg.visit([&](auto output) { vec.assign(output.begin(), output.end()); });
            rets.push_back(vec);
        }

        return rets;
    };

    // then branch
    {
        auto results = run_prog(true);
        std::vector<float> gold0(4, 2.0f);
        std::vector<float> gold1(12, 4.0f);
        EXPECT(migraphx::verify_range(results.at(0), gold0));
        EXPECT(migraphx::verify_range(results.at(1), gold1));
    }

    // else branch
    {
        auto results = run_prog(false);
        std::vector<float> gold0(4, 3.0f);
        std::vector<float> gold1(12, 5.0f);
        EXPECT(migraphx::verify_range(results.at(0), gold0));
        EXPECT(migraphx::verify_range(results.at(1), gold1));
    }
}

TEST_CASE(instance_norm_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_test.onnx");

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector(9);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.54919,
                               -1.16189,
                               -0.774596,
                               -0.387298,
                               0,
                               0.387298,
                               0.774596,
                               1.16189,
                               1.54919,
                               -2.09838,
                               -1.32379,
                               -0.549192,
                               0.225404,
                               1,
                               1.7746,
                               2.54919,
                               3.32379,
                               4.09838};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(instance_norm_3d_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_3d_test.onnx");

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector(16);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.52752,
                               -1.09109,
                               -0.654653,
                               -0.218218,
                               0.218218,
                               0.654653,
                               1.09109,
                               1.52752,
                               -2.05505,
                               -1.18218,
                               -0.309306,
                               0.563565,
                               1.43644,
                               2.30931,
                               3.18218,
                               4.05505};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(lessorequal_test)
{
    migraphx::program p = migraphx::parse_onnx("lessorequal_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1 = {0.25, 0.75, 0.9375};
    std::vector<float> data2 = {0.25, 0.74, 0.9411};

    migraphx::parameter_map pp;
    pp["x1"] = migraphx::argument(s, data1.data());
    pp["x2"] = migraphx::argument(s, data2.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 0, 1};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_downsample_f_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_downsample_f_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.0f);

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0f, 3.0f};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_upsample_linear_ac_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_ac_test.onnx");
    p.compile(migraphx::ref::target{});

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

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_upsample_linear_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_upsample_pf_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_pf_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                               3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(selu_test)
{
    migraphx::program p = migraphx::parse_onnx("selu_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape xs{migraphx::shape::double_type, {2, 3}};
    std::vector<double> x_data = {1.1, 2.1, 0.0, -1.3, -5.3, 12.0};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.55, 1.05, 0, -0.10912, -0.149251, 6};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {3, 2}};
    std::vector<float> data = {0, 1, 2, 3, 4, 5};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2, 3};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_5arg_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {10, 11, 12, 13, 15, 16, 17, 18};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_reverse_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_reverse_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {14, 13, 12, 11, 19, 18, 17, 16};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_step_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_step_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {13, 11};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(upsample_test)
{
    migraphx::program p = migraphx::parse_onnx("upsample_test.onnx");

    std::vector<float> x_data = {1, 2, 3, 4};
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                               3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(where_test)
{
    migraphx::program p = migraphx::parse_onnx("where_test.onnx");
    p.compile(migraphx::ref::target{});

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
    EXPECT(migraphx::verify_range(result_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
