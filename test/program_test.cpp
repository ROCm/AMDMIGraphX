
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/ref/target.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(sum_op{}, x, y);
    auto one = mm->add_literal(1);
    mm->add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(program_equality)
{
    migraphx::program x = create_program();
    migraphx::program y = create_program();
    EXPECT(x == y);
}

TEST_CASE(program_not_equality1)
{
    migraphx::program x;
    migraphx::program y = create_program();
    EXPECT(x != y);
    x = y;
    EXPECT(x == y);
}

TEST_CASE(program_not_equality2)
{
    migraphx::program x;
    migraphx::program y = create_program();
    EXPECT(x != y);
    y = x;
    EXPECT(x == y);
}

TEST_CASE(program_default_copy_construct)
{
    migraphx::program x;
    migraphx::program y;
    EXPECT(x == y);
}

TEST_CASE(program_copy)
{
    auto create_program_1 = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {3, 4, 5}};
        std::vector<float> data(3 * 4 * 5);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l2  = mm->add_literal(migraphx::literal(s, data));
        auto p1  = mm->add_parameter("x", s);
        auto po  = mm->add_outline(s);
        auto sum = mm->add_instruction(migraphx::op::add{}, l2, po);
        mm->add_instruction(migraphx::op::mul{}, sum, p1);

        return p;
    };

    {
        auto p1 = create_program_1();
        migraphx::program p2{};
        p2 = p1;

        p2.compile(migraphx::ref::target{});
        EXPECT(p1 != p2);

        p1.compile(migraphx::ref::target{});
        EXPECT(p1 == p2);

        EXPECT(p1.get_parameter_names() == p2.get_parameter_names());
    }

    {
        auto p1 = create_program_1();
        auto p2(p1);
        EXPECT(p1 == p2);

        p1.compile(migraphx::ref::target{});
        EXPECT(p1 != p2);

        p2 = p1;
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_1();
        auto p2 = create_program();
        EXPECT(p1 != p2);

        p2 = p1;
        EXPECT(p1 == p2);

        p1.compile(migraphx::ref::target{});
        p2.compile(migraphx::ref::target{});

        EXPECT(p1 == p2);
    }

    {
        migraphx::program p1;
        auto* mm1 = p1.get_main_module();

        migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};
        migraphx::shape s3{migraphx::shape::float_type, {2, 6}};
        auto para1 = mm1->add_parameter("m1", s1);
        auto para2 = mm1->add_parameter("m2", s2);
        auto para3 = mm1->add_parameter("m3", s3);
        mm1->add_instruction(migraphx::op::dot{0.31f, 0.28f}, para1, para2, para3);

        migraphx::program p2{};
        p2 = p1;
        EXPECT(p2 == p1);

        p1.compile(migraphx::ref::target{});
        p2.compile(migraphx::ref::target{});
        EXPECT(p2 == p1);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
