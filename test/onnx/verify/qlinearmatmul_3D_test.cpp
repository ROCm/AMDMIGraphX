
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearmatmul_3D_test)
{
    // https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearMatMul.html

    migraphx::program p = migraphx::parse_onnx("qlinearmatmul_3D_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {2, 2, 4}};
    std::vector<uint8_t> data_a = {
        208, 236, 0, 238, 3, 214, 255, 29, 208, 236, 0, 238, 3, 214, 255, 29};

    migraphx::shape b{migraphx::shape::uint8_type, {2, 4, 3}};
    std::vector<uint8_t> data_b = {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247,
                                   152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {168, 115, 255, 1, 66, 151, 168, 115, 255, 1, 66, 151};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
