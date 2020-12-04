#include <migraphx/auto_contiguous.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(*p.get_main_module(), {migraphx::auto_contiguous{}});
}

// TODO: Add this test case
void literal_broadcast()
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    mm->add_literal(get_2_broadcasted());
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().broadcasted());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().broadcasted());
}

TEST_CASE(literal_transpose)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    mm->add_literal(get_2x2_transposed());
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
}

TEST_CASE(after_literal_transpose)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    auto t = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    mm->add_instruction(pass_op{}, t);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
}

TEST_CASE(after_literal_broadcast)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l1  = mm->add_literal(get_2x2());
    auto l2  = mm->add_literal(get_2());
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().broadcasted());
    auto b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"dims", l1->get_shape().lens()}}), l2);
    mm->add_instruction(pass_op{}, b);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().broadcasted());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().broadcasted());
}

TEST_CASE(after_param_transpose)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
    auto t = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    mm->add_instruction(pass_op{}, t);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().transposed());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().transposed());
}

TEST_CASE(after_param_broadcast)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l1  = mm->add_parameter("2x2", {migraphx::shape::float_type, {2, 2}});
    auto l2  = mm->add_parameter("2", {migraphx::shape::float_type, {2}});
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().broadcasted());
    auto b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 0}, {"dims", l1->get_shape().lens()}}), l2);
    mm->add_instruction(pass_op{}, b);
    EXPECT(not p.get_output_shapes().back().standard());
    EXPECT(p.get_output_shapes().back().broadcasted());
    run_pass(p);
    EXPECT(p.get_output_shapes().back().standard());
    EXPECT(not p.get_output_shapes().back().broadcasted());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
