
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(slice_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_data{migraphx::shape::float_type, {3, 2}};
    std::vector<float> data = {0, 1, 2, 3, 4, 5};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2, 3};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
