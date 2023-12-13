
#include <onnx_test.hpp>


TEST_CASE(mean_integral_test)
{
    const std::size_t num_data = 10;
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 2}};

    auto mean = mm->add_parameter("0", s);
    for(std::size_t i = 1; i < num_data; ++i)
    {
        auto data = mm->add_parameter(std::to_string(i), s);
        mean      = mm->add_instruction(migraphx::make_op("add"), mean, data);
    }

    auto div_lit = mm->add_literal(migraphx::literal{migraphx::shape{s.type()}, {num_data}});
    auto divisor =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), div_lit);
    mean = mm->add_instruction(migraphx::make_op("div"), mean, divisor);

    auto prog = optimize_onnx("mean_integral_test.onnx");

    EXPECT(p == prog);
}


