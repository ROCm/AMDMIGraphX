
#include <onnx_test.hpp>

TEST_CASE(gathernd_batch_dims_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto l1  = mm->add_parameter("indices", migraphx::shape{migraphx::shape::int64_type, {2, 1}});
    int batch_dims = 1;
    mm->add_instruction(migraphx::make_op("gathernd", {{"batch_dims", batch_dims}}), l0, l1);
    auto prog = optimize_onnx("gathernd_batch_dims_test.onnx");

    EXPECT(p == prog);
}
