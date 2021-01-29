#include <migraphx/auto_contiguous.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m) { migraphx::run_passes(m, {migraphx::auto_contiguous{}}); }

// TODO: Add this test case
void literal_broadcast()
{
    migraphx::module m;

    m.add_literal(get_2_broadcasted());
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().broadcasted());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
}

TEST_CASE(literal_transpose)
{
    migraphx::module m;

    m.add_literal(get_2x2_transposed());
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().transposed());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
}

TEST_CASE(after_literal_transpose)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    m.add_instruction(pass_op{}, t);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().transposed());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
}

TEST_CASE(after_literal_broadcast)
{
    migraphx::module m;

    auto l1 = m.add_literal(get_2x2());
    auto l2 = m.add_literal(get_2());
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
    auto b = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"dims", l1->get_shape().lens()}}), l2);
    m.add_instruction(pass_op{}, b);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().broadcasted());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
}

TEST_CASE(after_param_transpose)
{
    migraphx::module m;

    auto l = m.add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    m.add_instruction(pass_op{}, t);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().transposed());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().transposed());
}

TEST_CASE(after_param_broadcast)
{
    migraphx::module m;

    auto l1 = m.add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    auto l2 = m.add_parameter("2", {migraphx::shape::float_type, {2}});
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
    auto b = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"dims", l1->get_shape().lens()}}), l2);
    m.add_instruction(pass_op{}, b);
    EXPECT(not m.get_output_shapes().back().standard());
    EXPECT(m.get_output_shapes().back().broadcasted());
    run_pass(m);
    EXPECT(m.get_output_shapes().back().standard());
    EXPECT(not m.get_output_shapes().back().broadcasted());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
