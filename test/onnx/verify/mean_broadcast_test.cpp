
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(mean_broadcast_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_broadcast_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> data0(12, 1);
    migraphx::shape s1{migraphx::shape::float_type, {1, 2, 3, 4}};
    std::vector<float> data1(24, 2);
    migraphx::shape s2{migraphx::shape::float_type, {4}};
    std::vector<float> data2(4, 3);
    migraphx::shape s3{migraphx::shape::float_type, {1}};
    std::vector<float> data3(1, 4);
    migraphx::shape s4{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> data4(6, 5);

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(s0, data0.data());
    pp["1"] = migraphx::argument(s1, data1.data());
    pp["2"] = migraphx::argument(s2, data2.data());
    pp["3"] = migraphx::argument(s3, data3.data());
    pp["4"] = migraphx::argument(s4, data4.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(24, 3);
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


