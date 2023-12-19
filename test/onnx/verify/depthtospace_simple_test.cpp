
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(depthtospace_simple_test)
{
    auto p = migraphx::parse_onnx("depthtospace_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data_in(48);
    std::iota(std::begin(data_in), std::end(data_in), 0);
    migraphx::shape s_x{migraphx::shape::float_type, {1, 8, 2, 3}};
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, data_in.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0,  12, 1,  13, 2,  14, 24, 36, 25, 37, 26, 38, 3,  15, 4,  16,
                               5,  17, 27, 39, 28, 40, 29, 41, 6,  18, 7,  19, 8,  20, 30, 42,
                               31, 43, 32, 44, 9,  21, 10, 22, 11, 23, 33, 45, 34, 46, 35, 47};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


