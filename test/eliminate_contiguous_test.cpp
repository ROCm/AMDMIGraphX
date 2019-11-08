#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/sin.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/contiguous.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::eliminate_contiguous{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(standard_op)
{
    migraphx::program p;
    auto l = p.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_standard_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(standard_op_const)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_standard_op{}, c);
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
}

TEST_CASE(non_standard_op)
{
    migraphx::program p;
    auto l = p.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(non_standard_op_const)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_op{}, c);
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == 2);
}

TEST_CASE(transpose_gemm)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t  = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c  = p.add_instruction(migraphx::op::contiguous{}, t);
    auto ic = p.add_instruction(migraphx::op::identity{}, c);
    p.add_instruction(migraphx::op::dot{}, ic, l);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
}

TEST_CASE(transpose_standard_op)
{
    migraphx::program p;
    auto l  = p.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t  = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c  = p.add_instruction(migraphx::op::contiguous{}, t);
    auto sn = p.add_instruction(migraphx::op::sin{}, c);
    p.add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(transpose_standard_op_const)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t  = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c  = p.add_instruction(migraphx::op::contiguous{}, t);
    auto sn = p.add_instruction(migraphx::op::sin{}, c);
    p.add_instruction(pass_standard_op{}, sn);
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == 3);
}

TEST_CASE(no_packed_unary_op)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t  = p.add_instruction(migraphx::op::slice{{1}, {1}, {2}}, l);
    auto c  = p.add_instruction(migraphx::op::contiguous{}, t);
    auto sn = p.add_instruction(migraphx::op::sin{}, c);
    p.add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(p.begin(), p.end());
    run_pass(p);
    EXPECT(std::distance(p.begin(), p.end()) == count - 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
