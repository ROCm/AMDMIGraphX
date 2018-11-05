#include <migraph/eliminate_concat.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct concat
{
    concat(std::size_t axis) { op.axis = axis; }
    migraph::op::concat op;
    std::string name() const { return "eliminate_concat::concat"; }
    migraph::shape compute_shape(std::vector<migraph::shape> inputs) const
    {
        return op.compute_shape(inputs);
    }
    migraph::argument compute(migraph::context& ctx,
                              const migraph::shape& output_shape,
                              const std::vector<migraph::argument>& args) const
    {
        return {output_shape};
    }
};

struct concat_test_optimization
{
    /// A unique name used to identify the concat optimization
    std::string name() const { return "eliminate_concat::concat"; }
    /// A unique name used to identify the allocate operator
    std::string allocate() const { return "allocate"; }
    /// Return the lowered concat operator
    migraph::op::concat get_concat(const migraph::operation& op) const
    {
        return migraph::any_cast<concat>(op).op;
    }
};

struct eliminate_concat_target
{
    std::size_t align = 32;
    std::string name() const { return "eliminate_target"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::eliminate_concat{concat_test_optimization{}},
                migraph::dead_code_elimination{}};
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

struct fred_op
{
    std::string name() const { return "fred_op"; }
    migraph::shape compute_shape(const std::vector<migraph::shape>& inputs) const
    {
        migraph::check_shapes{inputs}.has(1);
        return inputs.at(0);
    }
    migraph::argument compute(migraph::context&,
                              const migraph::shape& output_shape,
                              const std::vector<migraph::argument>& args) const
    {
        return args.at(0);
    }
};

void basic()
{
    auto create_test_program = []() {
        migraph::program p;
        auto a1 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1, 2, 8, 8}}});
        auto p1 = p.add_instruction(fred_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1, 3, 8, 8}}});
        auto p2 = p.add_instruction(fred_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1, 5, 8, 8}}});
        auto p3          = p.add_instruction(fred_op{}, a3);
        std::size_t axis = 1;
        auto a4 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1, 10, 8, 8}}});
        auto p4 = p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };
    auto create_control_program = []() {
        migraph::program p;
        auto a1 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {1, 10, 8, 8}}});
        auto l1 = p.add_instruction(
            migraph::op::load{migraph::shape{migraph::shape::float_type, {1, 2, 8, 8}}, 0}, {a1});
        auto p1 = p.add_instruction(fred_op{}, l1);
        auto l2 = p.add_instruction(
            migraph::op::load{migraph::shape{migraph::shape::float_type, {1, 3, 8, 8}}, 512}, {a1});
        auto p2 = p.add_instruction(fred_op{}, l2);
        auto l3 = p.add_instruction(
            migraph::op::load{migraph::shape{migraph::shape::float_type, {1, 5, 8, 8}}, 1280},
            {a1});
        auto p3 = p.add_instruction(fred_op{}, l3);
        auto i1 = p.add_instruction(migraph::op::identity{}, {a1, p1, p2, p3});
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    p1.compile(eliminate_concat_target{});

    EXPECT(p1 == p2);
}

void wont_work()
{
    auto create_test_program = []() {
        migraph::program p;
        auto a1 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 2, 8, 8}}});
        auto p1 = p.add_instruction(fred_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 3, 8, 8}}});
        auto p2 = p.add_instruction(fred_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = p.add_instruction(fred_op{}, a3);
        std::size_t axis = 1;
        auto a4 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 10, 8, 8}}});
        auto p4 = p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };
    auto create_control_program = []() {
        migraph::program p;
        auto a1 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 2, 8, 8}}});
        auto p1 = p.add_instruction(fred_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 3, 8, 8}}});
        auto p2 = p.add_instruction(fred_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = p.add_instruction(fred_op{}, a3);
        std::size_t axis = 1;
        auto a4 =
            p.add_instruction(allocate{migraph::shape{migraph::shape::float_type, {2, 10, 8, 8}}});
        auto p4 = p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    p1.compile(eliminate_concat_target{});

    EXPECT(p1 == p2);
}

int main()
{
    setenv("MIGRAPH_DISABLE_MEMORY_COLORING", "1", 1);
    basic();
    wont_work();
}
