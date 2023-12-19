
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(slice_5arg_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {10, 11, 12, 13, 15, 16, 17, 18};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


