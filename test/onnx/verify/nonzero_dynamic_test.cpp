
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(nonzero_test)
{
    migraphx::program p = migraphx::parse_onnx("nonzero_dynamic_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::bool_type, {2, 2}};
    std::vector<char> data = {1, 1, 1, 0};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


