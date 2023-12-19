
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(lessorequal_test)
{
    migraphx::program p = migraphx::parse_onnx("lessorequal_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


