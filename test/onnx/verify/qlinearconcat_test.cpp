
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearconcat_test)
{
    auto p = migraphx::parse_onnx("qlinearconcat_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<int8_t> data_t0 = {2, 3};
    migraphx::shape s_t0{migraphx::shape::int8_type, {2}};
    migraphx::parameter_map pp;
    pp["t0"] = migraphx::argument(s_t0, data_t0.data());

    std::vector<int8_t> data_t1 = {6, 8, 10};
    migraphx::shape s_t1{migraphx::shape::int8_type, {3}};
    pp["t1"] = migraphx::argument(s_t1, data_t1.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {3, 4, 5, 6, 7};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
