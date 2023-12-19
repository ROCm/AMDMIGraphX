
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearmatmul_2D_test)
{
    migraphx::program p = migraphx::parse_onnx("qlinearmatmul_2D_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {1, 8}};
    std::vector<uint8_t> data_a = {2, 4, 6, 8, 10, 12, 14, 16};

    migraphx::shape b{migraphx::shape::uint8_type, {8, 1}};
    std::vector<uint8_t> data_b = {126, 130, 124, 132, 122, 134, 120, 136};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {66};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
