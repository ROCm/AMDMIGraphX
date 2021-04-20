#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::eliminate_contiguous{"contiguous"}, migraphx::dead_code_elimination{}});
}

TEST_CASE(standard_op)
{
    migraphx::module m;

    auto l = m.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_standard_op{}, c);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(standard_op_const)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_standard_op{}, c);
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == 2);
}

TEST_CASE(non_standard_op)
{
    migraphx::module m;

    auto l = m.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_op{}, c);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(non_standard_op_const)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    auto t = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c = m.add_instruction(migraphx::make_op("contiguous"), t);
    m.add_instruction(pass_op{}, c);
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == 2);
}

TEST_CASE(transpose_gem)
{
    migraphx::module m;

    auto l  = m.add_literal(get_2x2());
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto ic = m.add_instruction(migraphx::make_op("identity"), c);
    m.add_instruction(migraphx::make_op("dot"), ic, l);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == (count - 1));
}

TEST_CASE(transpose_standard_op)
{
    migraphx::module m;

    auto l  = m.add_parameter("x", {migraphx::shape::float_type, {2, 2}});
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = m.add_instruction(migraphx::make_op("sin"), c);
    m.add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

TEST_CASE(transpose_standard_op_const)
{
    migraphx::module m;

    auto l  = m.add_literal(get_2x2());
    auto t  = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = m.add_instruction(migraphx::make_op("sin"), c);
    m.add_instruction(pass_standard_op{}, sn);
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == 3);
}

TEST_CASE(no_packed_unary_op)
{
    migraphx::module m;

    auto l = m.add_literal(get_2x2());
    auto t = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), t);
    auto sn = m.add_instruction(migraphx::make_op("sin"), c);
    m.add_instruction(pass_standard_op{}, sn);
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count - 1);
}

TEST_CASE(non_standard_return_input)
{
    migraphx::module m;

    auto l  = m.add_literal(get_2x2());
    auto tl = m.add_instruction(migraphx::make_op("transpose", {{"dims", {1, 0}}}), l);
    auto c  = m.add_instruction(migraphx::make_op("contiguous"), tl);
    m.add_return({c});
    auto count = std::distance(m.begin(), m.end());
    run_pass(m);
    EXPECT(std::distance(m.begin(), m.end()) == count);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
