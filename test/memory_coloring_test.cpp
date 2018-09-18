#include <migraph/memory_coloring.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct memory_coloring_target
{
    std::string name() const { return "memory_coloring"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::memory_coloring{"allocate"}};
    }
    migraph::context get_context() const { return {}; }
};

struct allocate
{
    migraph::shape s{};
    std::string name() const { return "allocate"; }
    migraph::shape compute_shape(const std::vector<migraph::shape>& inputs) const
    {
        migraph::check_shapes{inputs}.has(0);
        return s;
    }
    migraph::argument compute(migraph::context&,
                              const migraph::shape& output_shape,
                              const std::vector<migraph::argument>&) const
    {
        return {output_shape};
    }
};

int main()
{
    migraph::program p;
    auto a1 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {8}}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {40}}});
    p.add_instruction(pass_op{}, a2, p1);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 192);
}
