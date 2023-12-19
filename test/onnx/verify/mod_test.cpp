
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(mod_test)
{
    migraphx::program p = migraphx::parse_onnx("mod_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::int32_type, {3, 3, 3}};

    std::vector<int32_t> a = {-4, 7, 5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7,
                              5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7, 5};

    std::vector<int32_t> b = {2, -3, 8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3,
                              8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3, 8};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s, a.data());
    p_map["1"] = migraphx::argument(s, b.data());

    auto result = p.eval(p_map).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold = {0, -2, 5, 0, 2, 3,  0, -2, 5, 0, 2, 3,  0, -2,
                                 5, 0,  2, 3, 0, -2, 5, 0,  2, 3, 0, -2, 5};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


