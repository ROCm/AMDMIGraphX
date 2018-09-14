#include <migraph/memory_coloring.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct memory_coloring_target
{
    std::string name() const { return "memory_coloring"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::memory_coloring{}};
    }
    migraph::context get_context() const { return {}; }
};

int main()
{
    migraph::program p;
    auto l = p.add_literal(get_2x2());
    p.add_instruction(migraph::transpose{{1, 0}}, l);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 16);
}

