#include <migraphx/memory_coloring.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::memory_coloring{"allocate", true}});
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
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return {output_shape};
    }
};

migraphx::instruction_ref add_alloc(migraphx::program& p, const migraphx::shape& s)
{
    return p.add_instruction(allocate{s});
}

bool no_allocate(const migraphx::program& p)
{
    return std::none_of(p.begin(), p.end(), [](auto&& ins) { return ins.name() == "allocate"; });
}

TEST_CASE(test1)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test2)
{
    migraphx::program p;
    auto input = p.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {16}});

    auto a1 = add_alloc(p, {migraphx::shape::float_type, {128}});
    auto p1 = p.add_instruction(pass_op{}, a1, input);
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(p));
}

TEST_CASE(test3)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {128}});
    auto p1 = p.add_instruction(pass_op{}, p2, a1);
    auto p3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p3, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(p));
}

TEST_CASE(test4)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {128}});
    auto p1 = p.add_instruction(pass_op{}, p2, a1);
    auto p3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p3, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(p));
}

TEST_CASE(test5)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test6)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

TEST_CASE(test7)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

TEST_CASE(test8)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(p, {migraphx::shape::float_type, {192}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 960);
    CHECK(no_allocate(p));
}

TEST_CASE(test9)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 96);
    CHECK(no_allocate(p));
}

TEST_CASE(test10)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 32);
    CHECK(no_allocate(p));
}

TEST_CASE(test11)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

TEST_CASE(test12)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

TEST_CASE(test13)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

TEST_CASE(test14)
{
    migraphx::program p;
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

TEST_CASE(test15)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a3, p1, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

TEST_CASE(test16)
{
    migraphx::program p;
    auto a1 = p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {40}}));
    auto p2 = p.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a3, p1, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(p));
}

TEST_CASE(test17)
{
    migraphx::program p;
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a1 = p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {40}}));
    auto p2 = p.add_instruction(pass_op{}, a2);
    p.add_instruction(pass_op{}, a3, p1, p2);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(p));
}

TEST_CASE(test18)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = p.add_instruction(pass_op{}, a1, p1);
    auto p3 = p.add_instruction(pass_op{}, p2, p1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1, p2, p3);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test19)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a3, p2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

TEST_CASE(test20)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraphx::shape::float_type, {32}});
    p.add_instruction(pass_op{}, a4, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(p));
}

TEST_CASE(test21)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(p));
}

TEST_CASE(test22)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(p));
}

TEST_CASE(test23)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(p));
}

TEST_CASE(test24)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(p));
}

TEST_CASE(test25)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(nop{});
    auto p1 = p.add_instruction(pass_op{}, a1);
    p.add_instruction(nop{});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test26)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(nop{}, a1);
    auto p1 = p.add_instruction(pass_op{}, a1);
    p.add_instruction(nop{}, a1, p1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test27)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(nop{}, a2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test28)
{
    migraphx::program p;
    auto output = p.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1     = p.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2     = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, p2, output);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test29)
{
    migraphx::program p;
    auto output = p.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1     = p.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2     = p.add_instruction(pass_op{}, a2, p1);
    p.move_instruction(output, p2);
    p.add_instruction(pass_op{}, p2, output);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test30)
{
    migraphx::program p;
    auto output = p.add_parameter("x", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1     = p.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2     = p.add_instruction(pass_op{}, a2, p1);
    p.move_instruction(output, p2);
    p.add_instruction(pass_op{}, p2, output);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test31)
{
    migraphx::program p;
    auto output = p.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1     = p.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.move_instruction(output, a2);
    p.add_instruction(pass_op{}, a2, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test32)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p1 = p.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a5, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

TEST_CASE(test33)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(p, {migraphx::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a5, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

TEST_CASE(test34)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p1 = p.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a5, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 480);
    CHECK(no_allocate(p));
}

TEST_CASE(test35)
{
    migraphx::program p;
    auto a1 = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto a2 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraphx::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(p, {migraphx::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a5, p1);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

TEST_CASE(test36)
{
    migraphx::program p;
    auto output = p.add_parameter("output", {migraphx::shape::float_type, {20}});
    auto a1     = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto a2     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p1     = p.add_instruction(pass_op{}, a2, a1);
    auto a3     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2     = p.add_instruction(pass_op{}, a3, p1);
    auto a4     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p3     = p.add_instruction(pass_op{}, a4, p2);
    p.add_instruction(pass_op{}, output, p3);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 320);
    CHECK(no_allocate(p));
}

TEST_CASE(test37)
{
    migraphx::program p;
    auto output = p.add_parameter("output", {migraphx::shape::float_type, {20}});
    auto a1     = add_alloc(p, {migraphx::shape::float_type, {4}});
    auto a2     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p1     = p.add_instruction(pass_op{}, a2, a1);
    auto a3     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p2     = p.add_instruction(pass_op{}, a3, p1);
    auto a4     = add_alloc(p, {migraphx::shape::float_type, {40}});
    auto p3     = p.add_instruction(pass_op{}, a4, p2);
    p.add_instruction(pass_op{}, output, p3);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 320);
    CHECK(no_allocate(p));
}

TEST_CASE(test38)
{
    migraphx::program p;
    auto output = p.add_parameter("output", {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p29    = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto p30    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p31    = p.add_instruction(pass_op{}, p30, p29);
    auto p32    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p37    = p.add_instruction(pass_op{}, p32, p31);
    auto p38    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p39    = p.add_instruction(pass_op{}, p38, p37);
    auto p40    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p41    = p.add_instruction(pass_op{}, p40, p39);
    auto p42    = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto p43    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p44    = p.add_instruction(pass_op{}, p43, p41, p42);
    auto p45    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p50    = p.add_instruction(pass_op{}, p45, p44);
    auto p51    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p52    = p.add_instruction(pass_op{}, p51, p50);
    auto p53    = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto p54    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p55    = p.add_instruction(pass_op{}, p54, p52, p53);
    auto p56    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p61    = p.add_instruction(pass_op{}, p56, p55);
    auto p62    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p63    = p.add_instruction(pass_op{}, p62, p61, p41);
    auto p64    = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto p65    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p66    = p.add_instruction(pass_op{}, p65, p63, p64);
    auto p67    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p72    = p.add_instruction(pass_op{}, p67, p66);
    auto p73    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p74    = p.add_instruction(pass_op{}, p73, p72);
    auto p75    = add_alloc(p, {migraphx::shape::float_type, {0}});
    auto p76    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p77    = p.add_instruction(pass_op{}, p76, p74, p75);
    auto p78    = add_alloc(p, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p83    = p.add_instruction(pass_op{}, p78, p77);
    p.add_instruction(pass_op{}, output, p83, p63);
    run_pass(p);
    CHECK(p.get_parameter_shape("scratch").bytes() == 7225344); // Optimal solution is 6422528
    CHECK(no_allocate(p));
}

TEST_CASE(literal_test)
{
    migraphx::program p;
    auto lit = generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
    p.add_literal(lit);
    run_pass(p);
    auto result = p.eval({});
    CHECK(lit == result);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
