
#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(external_data_diff_path_test)
{
    migraphx::program p = create_external_data_prog();

    auto prog = optimize_onnx("ext_path/external_data_test.onnx");
    EXPECT(p == prog);
}
