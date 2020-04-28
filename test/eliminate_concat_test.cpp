#include <migraphx/eliminate_concat.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/op/identity.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct concat
{
    concat(std::size_t axis) { op.axis = axis; }
    migraphx::op::concat op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

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

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::eliminate_concat{concat_test_optimization{}},
                          migraphx::dead_code_elimination{}});
}

struct allocate
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

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

struct simple_op
{
    std::string name() const { return "simple_op"; }
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
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

template <class... Ts>
migraphx::shape create_shape(Ts... xs)
{
    return migraphx::shape{migraphx::shape::float_type, {std::size_t(xs)...}};
}

using load     = migraphx::op::load;
using identity = migraphx::op::identity;

TEST_CASE(simple)
{
    auto create_test_program = [] {
        migraphx::program p;
        auto a1          = p.add_instruction(allocate{create_shape(1)});
        auto p1          = p.add_instruction(simple_op{}, a1);
        auto a2          = p.add_instruction(allocate{create_shape(1)});
        auto p2          = p.add_instruction(simple_op{}, a2);
        std::size_t axis = 0;
        auto a3          = p.add_instruction(allocate{create_shape(2)});
        p.add_instruction(concat(axis), p1, p2, a3);
        return p;
    };
    auto create_control_program = [] {
        migraphx::program p;
        auto a1 = p.add_instruction(allocate{create_shape(2)});
        auto l1 = p.add_instruction(load{create_shape(1), 0}, a1);
        auto p1 = p.add_instruction(simple_op{}, l1);
        auto l2 = p.add_instruction(load{create_shape(1), 4}, a1);
        auto p2 = p.add_instruction(simple_op{}, l2);
        p.add_instruction(identity{}, a1, p1, p2);
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(reversed)
{
    auto create_test_program = [] {
        migraphx::program p;
        auto a1          = p.add_instruction(allocate{create_shape(1)});
        auto p1          = p.add_instruction(simple_op{}, a1);
        auto a2          = p.add_instruction(allocate{create_shape(1)});
        auto p2          = p.add_instruction(simple_op{}, a2);
        std::size_t axis = 0;
        auto a3          = p.add_instruction(allocate{create_shape(2)});
        p.add_instruction(concat(axis), p2, p1, a3);
        return p;
    };
    auto create_control_program = [] {
        migraphx::program p;
        auto a1 = p.add_instruction(allocate{create_shape(2)});
        auto l1 = p.add_instruction(load{create_shape(1), 4}, a1);
        auto p1 = p.add_instruction(simple_op{}, l1);
        auto l2 = p.add_instruction(load{create_shape(1), 0}, a1);
        auto p2 = p.add_instruction(simple_op{}, l2);
        p.add_instruction(identity{}, a1, p2, p1);
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(nested)
{
    auto concat_test_program = [](auto& p) {
        auto a1          = p.add_instruction(allocate{create_shape(1)});
        auto p1          = p.add_instruction(simple_op{}, a1);
        auto a2          = p.add_instruction(allocate{create_shape(1)});
        auto p2          = p.add_instruction(simple_op{}, a2);
        std::size_t axis = 0;
        auto a3          = p.add_instruction(allocate{create_shape(2)});
        return p.add_instruction(concat(axis), p1, p2, a3);
    };
    auto create_test_program = [&] {
        migraphx::program p;
        auto concat1     = concat_test_program(p);
        auto concat2     = concat_test_program(p);
        std::size_t axis = 0;
        auto a1          = p.add_instruction(allocate{create_shape(4)});
        p.add_instruction(concat(axis), concat1, concat2, a1);
        return p;
    };
    auto concat_control_program = [](auto& p, auto a1) {
        auto l1 = p.add_instruction(load{create_shape(1), 0}, a1);
        auto p1 = p.add_instruction(simple_op{}, l1);
        auto l2 = p.add_instruction(load{create_shape(1), 4}, a1);
        auto p2 = p.add_instruction(simple_op{}, l2);
        return p.add_instruction(identity{}, a1, p1, p2);
    };
    auto create_control_program = [&] {
        migraphx::program p;
        auto a1      = p.add_instruction(allocate{create_shape(4)});
        auto l1      = p.add_instruction(load{create_shape(2), 0}, a1);
        auto concat1 = concat_control_program(p, l1);
        auto l2      = p.add_instruction(load{create_shape(2), 8}, a1);
        auto concat2 = concat_control_program(p, l2);
        p.add_instruction(identity{}, a1, concat1, concat2);
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(basic)
{
    auto create_test_program = [] {
        migraphx::program p;
        auto a1 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 8}}});
        auto p1 = p.add_instruction(simple_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 3, 8, 8}}});
        auto p2 = p.add_instruction(simple_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1, 5, 8, 8}}});
        auto p3          = p.add_instruction(simple_op{}, a3);
        std::size_t axis = 1;
        auto a4          = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {1, 10, 8, 8}}});
        p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };
    auto create_control_program = [] {
        migraphx::program p;
        auto a1 = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {1, 10, 8, 8}}});
        auto l1 = p.add_instruction(
            load{migraphx::shape{migraphx::shape::float_type, {1, 2, 8, 8}}, 0}, {a1});
        auto p1 = p.add_instruction(simple_op{}, l1);
        auto l2 = p.add_instruction(
            load{migraphx::shape{migraphx::shape::float_type, {1, 3, 8, 8}}, 512}, {a1});
        auto p2 = p.add_instruction(simple_op{}, l2);
        auto l3 = p.add_instruction(
            load{migraphx::shape{migraphx::shape::float_type, {1, 5, 8, 8}}, 1280}, {a1});
        auto p3 = p.add_instruction(simple_op{}, l3);
        p.add_instruction(identity{}, {a1, p1, p2, p3});
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(wont_work)
{
    auto create_test_program = [] {
        migraphx::program p;
        auto a1 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 2, 8, 8}}});
        auto p1 = p.add_instruction(simple_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 3, 8, 8}}});
        auto p2 = p.add_instruction(simple_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = p.add_instruction(simple_op{}, a3);
        std::size_t axis = 1;
        auto a4          = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {2, 10, 8, 8}}});
        p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };
    auto create_control_program = [] {
        migraphx::program p;
        auto a1 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 2, 8, 8}}});
        auto p1 = p.add_instruction(simple_op{}, a1);
        auto a2 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 3, 8, 8}}});
        auto p2 = p.add_instruction(simple_op{}, a2);
        auto a3 =
            p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2, 5, 8, 8}}});
        auto p3          = p.add_instruction(simple_op{}, a3);
        std::size_t axis = 1;
        auto a4          = p.add_instruction(
            allocate{migraphx::shape{migraphx::shape::float_type, {2, 10, 8, 8}}});
        p.add_instruction(concat(axis), p1, p2, p3, a4);
        return p;
    };

    auto p1 = create_test_program();
    auto p2 = create_control_program();
    run_pass(p1);

    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
