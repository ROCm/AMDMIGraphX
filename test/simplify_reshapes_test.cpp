#include <migraph/simplify_reshapes.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct simplify_reshapes_target
{
    std::string name() const { return "simplify_reshapes"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::simplify_reshapes{}, migraph::dead_code_elimination{}};
    }
    migraph::context get_context(migraph::parameter_map) const { return {}; }
};

void double_contig()
{
    migraph::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraph::transpose{{1, 0}}, l);
    auto c1 = p.add_instruction(migraph::contiguous{}, t1);
    auto c2 = p.add_instruction(migraph::contiguous{}, c1);
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

void double_transpose()
{
    migraph::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraph::transpose{{1, 0}}, l);
    auto t2 = p.add_instruction(migraph::transpose{{1, 0}}, t1);
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

void double_transpose_contig()
{
    migraph::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraph::transpose{{1, 0}}, l);
    auto c1 = p.add_instruction(migraph::contiguous{}, t1);
    auto t2 = p.add_instruction(migraph::transpose{{1, 0}}, c1);
    auto c2 = p.add_instruction(migraph::contiguous{}, t2);
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

void single_transpose()
{
    migraph::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraph::transpose{{1, 0}}, l);
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

void double_transpose_sin_pass()
{
    migraph::program p;
    auto l  = p.add_literal(get_2x2());
    auto t1 = p.add_instruction(migraph::transpose{{1, 0}}, l);
    p.add_instruction(migraph::transpose{{1, 0}}, t1);
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

void single_transpose_sin_pass()
{
    migraph::program p;
    auto l = p.add_literal(get_2x2());
    p.add_instruction(migraph::transpose{{1, 0}}, l);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    p.compile(simplify_reshapes_target{});
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    EXPECT(std::distance(p.begin(), p.end()) == 2);
    auto result = p.eval({});
    EXPECT(result != get_2x2());
}

int main()
{
    double_contig();
    double_transpose();
    double_transpose_contig();
    single_transpose();
    double_transpose_sin_pass();
    single_transpose_sin_pass();
}
