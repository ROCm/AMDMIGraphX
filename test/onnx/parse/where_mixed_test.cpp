
#include <onnx_test.hpp>


TEST_CASE(where_mixed_test)
{
    //  mixture of static and dynamic input shapes is not supported
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    EXPECT(test::throws([&] { migraphx::parse_onnx("where_mixed_test.onnx", options); }));
}


