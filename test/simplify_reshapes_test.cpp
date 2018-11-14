#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct simplify_reshapes_target
{
    std::string name() const { return "simplify_reshapes"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::simplify_reshapes{}, migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(double_contig)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c1 = p.add_instruction(migraphx::op::contiguous{}, t1);
    auto c2 = p.add_instruction(migraphx::op::contiguous{}, c1);
    p.add_instruction(pass_op{}, c2);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, t1);
    p.add_instruction(pass_op{}, t2);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(double_transpose_contig)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    auto c1 = p.add_instruction(migraphx::op::contiguous{}, t1);
    auto t2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, c1);
    auto c2 = p.add_instruction(migraphx::op::contiguous{}, t2);
    p.add_instruction(pass_op{}, c2);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    p.add_instruction(pass_op{}, t1);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 3);
    auto result = p.eval({});
    EXPECT(result != get_2x2());
}

TEST_CASE(double_transpose_sin_pass)
{
    migraphx::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    p.add_instruction(migraphx::op::transpose{{1, 0}}, t1);
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    // std::cout << p << std::endl;
    // TODO: Fix this
    // EXPECT(std::distance(p.begin(), p.end()) == 1);
    auto result = p.eval({});
    EXPECT(result == get_2x2());
}

TEST_CASE(single_transpose_sin_pass)
{
    migraphx::program p;
    auto l = p.add_literal(get_2x2());
    p.add_instruction(migraphx::op::transpose{{1, 0}}, l);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result != get_2x2());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
