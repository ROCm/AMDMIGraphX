
#include <onnx_test.hpp>


TEST_CASE(mean_single_input_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto data0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 2, 3}});
    mm->add_return({data0});

    auto prog = migraphx::parse_onnx("mean_single_input_test.onnx");

    EXPECT(p == prog);
}


