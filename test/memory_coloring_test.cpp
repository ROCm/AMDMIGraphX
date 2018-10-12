#include <migraph/memory_coloring.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
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
        migraph::check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }
    migraph::argument compute(migraph::context&,
                              const migraph::shape& output_shape,
                              const std::vector<migraph::argument>&) const
    {
        return {output_shape};
    }
};

// A custom test operator that takes a single argument and an allocation
// This operator's output is an operand alias of argument 1
struct pass_memory
{
    std::string name() const { return "memory_coloring::pass_memory"; }
    migraph::shape compute_shape(const std::vector<migraph::shape>& inputs) const
    {
        migraph::check_shapes{inputs, *this}.has(2);
        return inputs.at(1);
    }
    migraph::argument compute(migraph::context&,
                              const migraph::shape&,
                              const std::vector<migraph::argument>& args) const
    {
        return args[1];
    }
};

// The previous existing test
void test1()
{
    migraph::program p;
    auto a0 = p.add_outline(migraph::shape{migraph::shape::float_type, {8}});
    auto a1 = p.add_instruction(allocate{}, a0);
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_outline(migraph::shape{migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(allocate{}, a2);
    p.add_instruction(pass_op{}, p2, p1);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 192);
}

// This test uses the pass_memory operator
void test2()
{
    migraph::program p;
    auto input = p.add_parameter("input", migraph::shape{migraph::shape::float_type, {16}});

    auto a0 = p.add_outline(migraph::shape{migraph::shape::float_type, {128}});
    auto a1 = p.add_instruction(allocate{}, a0);
    auto p1 = p.add_instruction(pass_memory{}, input, a1);
    auto a2 = p.add_outline(migraph::shape{migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(allocate{}, a2);
    p.add_instruction(pass_memory{}, p1, p2);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 672);
}

// This test uses the pass_memory operator with two memory allocation passed together.
// This is similar to allocations done for workspaces, that is one allocation is aliased and the
// other is just used
void test3()
{
    migraph::program p;
    auto a0 = p.add_outline(migraph::shape{migraph::shape::float_type, {8}});
    auto a1 = p.add_instruction(allocate{}, a0);
    auto a2 = p.add_outline(migraph::shape{migraph::shape::float_type, {128}});
    auto p2 = p.add_instruction(allocate{}, a2);
    auto p1 = p.add_instruction(pass_memory{}, a1, p2);
    auto a3 = p.add_outline(migraph::shape{migraph::shape::float_type, {40}});
    auto p3 = p.add_instruction(allocate{}, a3);
    p.add_instruction(pass_memory{}, p1, p3);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 704);
}

// Like the previous test, but this tests a zero workspace memory allocation
void test4()
{
    migraph::program p;
    auto a0 = p.add_outline(migraph::shape{migraph::shape::float_type, {0}});
    auto a1 = p.add_instruction(allocate{}, a0);
    auto a2 = p.add_outline(migraph::shape{migraph::shape::float_type, {128}});
    auto p2 = p.add_instruction(allocate{}, a2);
    auto p1 = p.add_instruction(pass_memory{}, a1, p2);
    auto a3 = p.add_outline(migraph::shape{migraph::shape::float_type, {40}});
    auto p3 = p.add_instruction(allocate{}, a3);
    p.add_instruction(pass_memory{}, p1, p3);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 672);
}

void literal_test()
{
    migraph::program p;
    auto lit = generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
    p.add_literal(lit);
    p.compile(memory_coloring_target{});
    auto result = p.eval({});
    EXPECT(lit == result);
}

void test5()
{
    migraph::program p;
    std::vector<float> a_data;
    migraph::shape a_shape{migraph::shape::float_type, {2, 2}};
    auto input = p.add_parameter("input", migraph::shape{migraph::shape::float_type, {16}});
    auto ll = p.add_literal(migraph::literal{a_shape, a_data});
    auto a0 = p.add_outline(migraph::shape{migraph::shape::float_type, {128}});
    auto a1 = p.add_instruction(allocate{}, a0);
    p.add_instruction(migraph::op::transpose{{1,0}}, ll);
    p.add_instruction(pass_memory{}, input, a1);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 528);
}

int main()
{
    test1();
    test2();
    test3();
    test4();
    literal_test();
    test5();
}
