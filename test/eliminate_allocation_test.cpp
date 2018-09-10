#include <migraph/eliminate_allocation.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct eliminate_allocation_target
{
    std::size_t align = 32;
    std::string name() const { return "eliminate_allocation"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::eliminate_allocation{"allocate", align}, migraph::dead_code_elimination{}};
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

void basic()
{
    migraph::program p;
    auto a1 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {8}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {40}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    p.compile(eliminate_allocation_target{});
    EXPECT(p.get_shape() == migraph::shape{migraph::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (8 * 4 + 40 * 4 + 200 * 4));
}

void aligned()
{
    migraph::program p;
    auto a1 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    p.compile(eliminate_allocation_target{});
    EXPECT(p.get_shape() == migraph::shape{migraph::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (32 + 32 + 200 * 4));
}

void unaligned()
{
    migraph::program p;
    auto a1 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    p.compile(eliminate_allocation_target{1});
    EXPECT(p.get_shape() == migraph::shape{migraph::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

void float_aligned()
{
    migraph::program p;
    auto a1 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    p.compile(eliminate_allocation_target{4});
    EXPECT(p.get_shape() == migraph::shape{migraph::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

int main()
{
    basic();
    aligned();
    unaligned();
    float_aligned();
}
