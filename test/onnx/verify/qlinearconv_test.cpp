
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>


TEST_CASE(qlinearconv_test)
{
    // https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearConv.html
    migraphx::program p = migraphx::parse_onnx("qlinearconv_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 7, 7}};
    std::vector<uint8_t> x_data = {255, 174, 162, 25,  203, 168, 58,  15,  59,  237, 95,  129, 0,
                                   64,  56,  242, 153, 221, 168, 12,  166, 232, 178, 186, 195, 237,
                                   162, 237, 188, 39,  124, 77,  80,  102, 43,  127, 230, 21,  83,
                                   41,  40,  134, 255, 154, 92,  141, 42,  148, 247};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {0,   81,  93,  230, 52,  87,  197, 240, 196, 18,  160, 126, 255,
                                 191, 199, 13,  102, 34,  87,  243, 89,  23,  77,  69,  60,  18,
                                 93,  18,  67,  216, 131, 178, 175, 153, 212, 128, 25,  234, 172,
                                 214, 215, 121, 0,   101, 163, 114, 213, 107, 8};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


