#include <migraphx/eliminate_concat.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct concat
{
    concat(std::size_t axis) { op.axis = axis; }
    migraphx::op::concat op;
    std::string name() const { return "eliminate_concat::concat"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return op.compute_shape(std::move(inputs));
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
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
    migraphx::op::concat get_concat(const migraphx::operation& op) const
    {
        return migraphx::any_cast<concat>(op).op;
    }
};

struct eliminate_concat_target
{
    std::size_t align = 32;
    std::string name() const { return "eliminate_target"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::eliminate_concat{concat_test_optimization{}},
                migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return {}; }
};

struct allocate
{
    migraphx::shape s{};
    std::string name() const { return "allocate"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return {output_shape};
    }
};

struct fred_op
{
    std::string name() const { return "fred_op"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs}.has(1);
        return inputs.at(0);
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>& args) const
    {
        return args.at(0);
    }
};

TEST_CASE(basic)
{
    auto create_test_program = []() {
        migraphx::program p;
        auto a1 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 8}}});
        auto p1 = p.add_instruction(fred_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 3, 8, 8}}});
        auto p2 = p.add_instruction(fred_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 5, 8, 8}}});
        auto p3          = p.add_instruction(fred_op{}, a3);
        std::size_t axis = 1;
        auto a4          = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {1, 10, 8, 8}}});
        p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };
    auto create_control_program = []() {
        migraphx::program p;
        auto a1 = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {1, 10, 8, 8}}});
        auto l1 = p.add_instruction(
            migraphx::op::load{migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 8}}, 0},
            {a1});
        auto p1 = p.add_instruction(fred_op{}, l1);
        auto l2 = p.add_instruction(
            migraphx::op::load{migraphx::shape{migraphx::shape::float_type, {1, 3, 8, 8}}, 512},
            {a1});
        auto p2 = p.add_instruction(fred_op{}, l2);
        auto l3 = p.add_instruction(
            migraphx::op::load{migraphx::shape{migraphx::shape::float_type, {1, 5, 8, 8}}, 1280},
            {a1});
        auto p3 = p.add_instruction(fred_op{}, l3);
        p.add_instruction(migraphx::op::identity{}, {a1, p1, p2, p3});
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    p1.compile(eliminate_concat_target{});

    EXPECT(p1 == p2);
}

TEST_CASE(wont_work)
{
    auto create_test_program = []() {
        migraphx::program p;
        auto a1 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 2, 8, 8}}});
        auto p1 = p.add_instruction(fred_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 3, 8, 8}}});
        auto p2 = p.add_instruction(fred_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = p.add_instruction(fred_op{}, a3);
        std::size_t axis = 1;
        auto a4          = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {2, 10, 8, 8}}});
        p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };
    auto create_control_program = []() {
        migraphx::program p;
        auto a1 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 2, 8, 8}}});
        auto p1 = p.add_instruction(fred_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 3, 8, 8}}});
        auto p2 = p.add_instruction(fred_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = p.add_instruction(fred_op{}, a3);
        std::size_t axis = 1;
        auto a4          = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {2, 10, 8, 8}}});
        p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    p1.compile(eliminate_concat_target{});

    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
