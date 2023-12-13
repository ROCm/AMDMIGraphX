
#include <onnx_test.hpp>

TEST_CASE(mean_test)
{
    const std::size_t num_data = 3;
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {1, 2, 3}};
    auto data0   = mm->add_parameter("0", s);
    auto data1   = mm->add_parameter("1", s);
    auto data2   = mm->add_parameter("2", s);
    auto div_lit = mm->add_literal(migraphx::literal{migraphx::shape{s.type()}, {num_data}});
    auto divisor =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), div_lit);
    auto mean = mm->add_instruction(migraphx::make_op("div"), data0, divisor);
    divisor =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), div_lit);
    data1 = mm->add_instruction(migraphx::make_op("div"), data1, divisor);
    mean  = mm->add_instruction(migraphx::make_op("add"), mean, data1);
    divisor =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), div_lit);
    data2 = mm->add_instruction(migraphx::make_op("div"), data2, divisor);
    mean  = mm->add_instruction(migraphx::make_op("add"), mean, data2);

    auto prog = optimize_onnx("mean_fp16_test.onnx");

    EXPECT(p == prog);
}
