
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(qlinearconv_scale_1D_test)
{
    // https:xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html

    migraphx::program p = migraphx::parse_onnx("qlinearconv_scale_1D_test.onnx");

    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 5, 5}};

    std::vector<uint8_t> x_data = {0,   11,  21,  32,  42,  53,  64,  74,  85,  96,  106, 117, 128,
                                   138, 149, 159, 170, 181, 191, 202, 212, 223, 234, 244, 255};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // # (1, 2, 3, 3) output tensor
    std::vector<int8_t> gold = {
        -43, -29, -15, 28, 42, 56, 99, 113, 127, -43, -29, -15, 28, 42, 56, 99, 113, 127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
