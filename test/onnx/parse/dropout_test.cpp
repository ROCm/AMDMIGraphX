
#include <onnx_test.hpp>


TEST_CASE(dropout_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 2, 2}});
    auto out   = mm->add_instruction(migraphx::make_op("identity"), input);
    migraphx::shape s{migraphx::shape::bool_type, {1, 3, 2, 2}};
    std::vector<int8_t> vec(s.elements(), 1);
    mm->add_literal(migraphx::literal(s, vec));
    mm->add_return({out});

    auto prog = migraphx::parse_onnx("dropout_test.onnx");
    EXPECT(p == prog);
}


