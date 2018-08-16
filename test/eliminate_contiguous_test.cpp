#include <migraph/eliminate_contiguous.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct eliminate_contiguous_target
{
    std::string name() const { return "eliminate_contiguous"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::eliminate_contiguous{}, migraph::dead_code_elimination{}};
    }
    migraph::context get_context() const { return {}; }
};

void standard_op()
{
    migraph::program p;
    auto l = p.add_literal(get_2x2());
    auto t = p.add_instruction(migraph::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraph::contiguous{}, t);
    p.add_instruction(pass_standard_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    p.compile(eliminate_contiguous_target{});
    EXPECT(std::distance(p.begin(), p.end()) == count);
}

void non_standard_op()
{
    migraph::program p;
    auto l = p.add_literal(get_2x2());
    auto t = p.add_instruction(migraph::transpose{{1, 0}}, l);
    auto c = p.add_instruction(migraph::contiguous{}, t);
    p.add_instruction(pass_op{}, c);
    auto count = std::distance(p.begin(), p.end());
    p.compile(eliminate_contiguous_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
}

int main()
{
    standard_op();
    non_standard_op();
}
