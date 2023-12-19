
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(if_then_else_multi_output_shapes_inlined_test)
{
    migraphx::program p =
        migraphx::parse_onnx("if_then_else_multi_output_shapes_inlined_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape x_data{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::shape y_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(x_data, data.data());
    pp["y"] = migraphx::argument(y_data, data.data());

    auto result_args = p.eval(pp);
    auto result      = result_args.front();
    auto result_b    = result_args.back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> result_vector_back;
    result_b.visit([&](auto output) { result_vector_back.assign(output.begin(), output.end()); });

    result_vector.insert(result_vector.end(), result_vector_back.begin(), result_vector_back.end());

    std::vector<float> gold = {
        1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375, 0.125, 1.50, -0.125, 0.250, -0.250, -1.125};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
