
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(spacetodepth_simple_test)
{
    auto p = migraphx::parse_onnx("spacetodepth_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data_in(48);
    std::iota(std::begin(data_in), std::end(data_in), 0);
    migraphx::shape s_x{migraphx::shape::float_type, {1, 2, 4, 6}};
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, data_in.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0,  2,  4,  12, 14, 16, 24, 26, 28, 36, 38, 40, 1,  3,  5,  13,
                               15, 17, 25, 27, 29, 37, 39, 41, 6,  8,  10, 18, 20, 22, 30, 32,
                               34, 42, 44, 46, 7,  9,  11, 19, 21, 23, 31, 33, 35, 43, 45, 47};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(spacetodepth_depthtospace_test)
{
    // space to depth
    auto p1 = migraphx::parse_onnx("spacetodepth_simple_test.onnx");
    p1.compile(migraphx::make_target("ref"));
    std::vector<float> gold_data_in(48);
    std::iota(std::begin(gold_data_in), std::end(gold_data_in), 0);
    migraphx::shape s_x_1{migraphx::shape::float_type, {1, 2, 4, 6}};
    migraphx::parameter_map pp1;
    pp1["x"]     = migraphx::argument(s_x_1, gold_data_in.data());
    auto result1 = p1.eval(pp1).back();
    // depth to space
    auto p2 = migraphx::parse_onnx("depthtospace_simple_test.onnx");
    p2.compile(migraphx::make_target("ref"));
    migraphx::parameter_map pp2;
    pp2["x"]     = result1;
    auto result2 = p2.eval(pp2).back();
    std::vector<float> result_vector2;
    result2.visit([&](auto output) { result_vector2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector2, gold_data_in));
}


