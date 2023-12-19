
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(mod_test_fmod_different_types)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_fmod_different_dtypes.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_float{migraphx::shape::float_type, {3, 3, 3}};
    migraphx::shape s_int{migraphx::shape::int32_type, {3, 3, 3}};

    std::vector<float> a = {1.2,  -2.2, 3.3,  4.1,   -5.4,  6.7,   7.8,  -8.4, 9.9,
                            10.7, 11.2, 12.3, 13.9,  -14.2, 15.8,  16.6, 17.9, 18.2,
                            19.0, 20.0, 21.0, -22.0, 23.0,  -24.0, 25.2, 26.3, 27.1};

    std::vector<int32_t> b = {30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                              16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s_float, a.data());
    p_map["1"] = migraphx::argument(s_int, b.data());

    auto result = p.eval(p_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{1.2,  -2.2, 3.3,  4.1,  -5.4,  6.7,  7.8, -8.4, 9.9,
                            10.7, 11.2, 12.3, 13.9, -14.2, 15.8, 1.6, 3.9,  5.2,
                            7.0,  9.0,  1.0,  -4.0, 7.0,   -3.0, 1.2, 1.3,  3.1};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
