
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(qlinearaveragepool_notset_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_notset_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {22};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


