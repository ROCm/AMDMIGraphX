#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/scalar.hpp>
#include <migraphx/op/mul.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(*p.get_main_module(),
                         {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(const_add)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    auto one  = mm1->add_literal(1);
    auto two  = mm1->add_literal(2);
    auto sum  = mm1->add_instruction(migraphx::make_op("add"), one, two);
    mm1->add_instruction(pass_op{}, sum);
    run_pass(p1);

    migraphx::program p2;
    auto* mm2  = p2.get_main_module();
    auto total = mm2->add_literal(3);
    mm2->add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add_parameter)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    auto one  = mm1->add_parameter("one", {migraphx::shape::int32_type, {1}});
    auto two  = mm1->add_literal(2);
    auto sum  = mm1->add_instruction(migraphx::make_op("add"), one, two);
    mm1->add_instruction(pass_op{}, sum);
    run_pass(p1);

    migraphx::program p2;
    auto* mm2  = p2.get_main_module();
    auto total = mm2->add_literal(3);
    mm2->add_instruction(pass_op{}, total);
    EXPECT(p1 != p2);
}

TEST_CASE(const_multiadd)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    auto one  = mm1->add_literal(1);
    auto two  = mm1->add_literal(2);
    auto sum1 = mm1->add_instruction(migraphx::make_op("add"), one, two);
    auto sum2 = mm1->add_instruction(migraphx::make_op("add"), sum1, two);
    mm1->add_instruction(pass_op{}, sum2);
    run_pass(p1);

    migraphx::program p2;
    auto* mm2  = p2.get_main_module();
    auto total = mm2->add_literal(5);
    mm2->add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add_mul)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    auto one  = mm1->add_literal(1);
    auto two  = mm1->add_literal(2);
    auto mul  = mm1->add_instruction(migraphx::make_op("mul"), two, two);
    auto sum1 = mm1->add_instruction(migraphx::make_op("add"), one, mul);
    auto sum2 = mm1->add_instruction(migraphx::make_op("add"), sum1, two);
    mm1->add_instruction(pass_op{}, sum2);
    run_pass(p1);

    migraphx::program p2;
    auto* mm2  = p2.get_main_module();
    auto total = mm2->add_literal(7);
    mm2->add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_add_scalar)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();
    auto one  = mm1->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                    mm1->add_literal(1));
    auto two  = mm1->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                    mm1->add_literal(2));
    auto sum  = mm1->add_instruction(migraphx::make_op("add"), one, two);
    mm1->add_instruction(pass_op{}, sum);
    run_pass(p1);

    migraphx::program p2;
    auto* mm2 = p2.get_main_module();
    auto total =
        mm2->add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 2}}, {3, 3, 3, 3}});
    mm2->add_instruction(pass_op{}, total);
    EXPECT(p1 == p2);
}

TEST_CASE(const_scalar)
{
    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        auto one = mm1->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                        mm1->add_literal(1));
        mm1->add_instruction(pass_op{}, one);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm2 = p2.get_main_module();
        auto one = mm2->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                        mm2->add_literal(1));
        mm2->add_instruction(pass_op{}, one);
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
