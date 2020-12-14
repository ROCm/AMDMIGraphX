#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(*p.get_main_module(),
                         {migraphx::eliminate_contiguous{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(standard_op)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    mm->add_instruction(pass_standard_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(standard_op_const)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    mm->add_instruction(pass_standard_op{}, c);
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
}

TEST_CASE(non_standard_op)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    mm->add_instruction(pass_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(non_standard_op_const)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    mm->add_instruction(pass_op{}, c);
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
}

TEST_CASE(transpose_gemm)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    auto ic  = mm->add_instruction(migraphx::make_op("identity"), c);
    mm->add_instruction(migraphx::make_op("dot"), ic, l);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
}

TEST_CASE(transpose_standard_op)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    auto sn  = mm->add_instruction(migraphx::make_op("sin"), c);
    mm->add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(transpose_standard_op_const)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t   = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), t);
    auto sn  = mm->add_instruction(migraphx::make_op("sin"), c);
    mm->add_instruction(pass_standard_op{}, sn);
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == 3);
}

TEST_CASE(no_packed_unary_op)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto t   = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), l);
    auto c  = mm->add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = mm->add_instruction(migraphx::make_op("sin"), c);
    mm->add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count - 1);
}

TEST_CASE(non_standard_return_input)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l   = mm->add_literal(get_2x2());
    auto tl  = mm->add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c   = mm->add_instruction(migraphx::make_op("contiguous"), tl);
    mm->add_return({c});
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
