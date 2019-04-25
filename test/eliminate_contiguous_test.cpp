#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/contiguous.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct eliminate_contiguous_target
{
    std::string name() const { return "eliminate_contiguous"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::eliminate_contiguous{}, migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(standard_op)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_standard_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    p.compile(eliminate_contiguous_target{});
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

TEST_CASE(non_standard_op)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    auto t = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraphx::op::contiguous{}, t);
    p.add_instruction(pass_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    p.compile(eliminate_contiguous_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
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
    p.compile(eliminate_contiguous_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
