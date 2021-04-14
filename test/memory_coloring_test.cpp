#include <migraphx/memory_coloring.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::memory_coloring{"allocate", true}});
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

migraphx::instruction_ref add_alloc(migraphx::module& m, const migraphx::shape& s)
{
    return m.add_instruction(allocate{s});
}

bool no_allocate(const migraphx::module& m)
{
    return std::none_of(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "allocate"; });
}

TEST_CASE(test1)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test2)
{
    migraphx::module m;

    auto input = m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {16}});

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {128}});
    auto m1 = m.add_instruction(pass_op{}, a1, input);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(m));
}

TEST_CASE(test3)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {128}});
    auto m1 = m.add_instruction(pass_op{}, m2, a1);
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, p3, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(m));
}

TEST_CASE(test4)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {128}});
    auto m1 = m.add_instruction(pass_op{}, m2, a1);
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, p3, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(m));
}

TEST_CASE(test5)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test6)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test7)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test8)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {192}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 960);
    CHECK(no_allocate(m));
}

TEST_CASE(test9)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto p3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, p3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 96);
    CHECK(no_allocate(m));
}

TEST_CASE(test10)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 32);
    CHECK(no_allocate(m));
}

TEST_CASE(test11)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test12)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test13)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test14)
{
    migraphx::module m;

    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, a3, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test15)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a3, m1, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test16)
{
    migraphx::module m;

    auto a1 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {40}}));
    auto m2 = m.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a3, m1, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(m));
}

TEST_CASE(test17)
{
    migraphx::module m;

    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a1 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {40}}));
    auto m2 = m.add_instruction(pass_op{}, a2);
    m.add_instruction(pass_op{}, a3, m1, m2);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(m));
}

TEST_CASE(test18)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto m2 = m.add_instruction(pass_op{}, a1, m1);
    auto p3 = m.add_instruction(pass_op{}, m2, m1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1, m2, p3);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test19)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a3, m2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test20)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {32}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(m));
}

TEST_CASE(test21)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(m));
}

TEST_CASE(test22)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(m));
}

TEST_CASE(test23)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(m));
}

TEST_CASE(test24)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {32}});
    auto m1 = m.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a4, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(m));
}

TEST_CASE(test25)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(nop{});
    auto m1 = m.add_instruction(pass_op{}, a1);
    m.add_instruction(nop{});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test26)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(nop{}, a1);
    auto m1 = m.add_instruction(pass_op{}, a1);
    m.add_instruction(nop{}, a1, m1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test27)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(nop{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test28)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a2, m1);
    m.add_instruction(pass_op{}, m2, output);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test29)
{
    migraphx::module m;
    auto output = m.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a2, m1);
    m.move_instruction(output, m2);
    m.add_instruction(pass_op{}, m2, output);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test30)
{
    migraphx::module m;

    auto output = m.add_parameter("x", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a2, m1);
    m.move_instruction(output, m2);
    m.add_instruction(pass_op{}, m2, output);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test31)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {8}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1     = m.add_instruction(pass_op{}, a1);
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.move_instruction(output, a2);
    m.add_instruction(pass_op{}, a2, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test32)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(m));
}

TEST_CASE(test33)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {40}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(m));
}

TEST_CASE(test34)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 480);
    CHECK(no_allocate(m));
}

TEST_CASE(test35)
{
    migraphx::module m;

    auto a1 = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto a2 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto a3 = add_alloc(m, {migraphx::shape::float_type, {8}});
    auto m1 = m.add_instruction(pass_op{}, a2, a1, a3);
    auto a5 = add_alloc(m, {migraphx::shape::float_type, {8}});
    m.add_instruction(pass_op{}, a5, m1);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(m));
}

TEST_CASE(test36)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {20}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1     = m.add_instruction(pass_op{}, a2, a1);
    auto a3     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a3, m1);
    auto a4     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3     = m.add_instruction(pass_op{}, a4, m2);
    m.add_instruction(pass_op{}, output, p3);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 320);
    CHECK(no_allocate(m));
}

TEST_CASE(test37)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {20}});
    auto a1     = add_alloc(m, {migraphx::shape::float_type, {4}});
    auto a2     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m1     = m.add_instruction(pass_op{}, a2, a1);
    auto a3     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto m2     = m.add_instruction(pass_op{}, a3, m1);
    auto a4     = add_alloc(m, {migraphx::shape::float_type, {40}});
    auto p3     = m.add_instruction(pass_op{}, a4, m2);
    m.add_instruction(pass_op{}, output, p3);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 320);
    CHECK(no_allocate(m));
}

TEST_CASE(test38)
{
    migraphx::module m;

    auto output = m.add_parameter("output", {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto m29    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p30    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p31    = m.add_instruction(pass_op{}, p30, m29);
    auto p32    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p37    = m.add_instruction(pass_op{}, p32, p31);
    auto p38    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 112, 112}});
    auto p39    = m.add_instruction(pass_op{}, p38, p37);
    auto p40    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p41    = m.add_instruction(pass_op{}, p40, p39);
    auto p42    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p43    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p44    = m.add_instruction(pass_op{}, p43, p41, p42);
    auto p45    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p50    = m.add_instruction(pass_op{}, p45, p44);
    auto p51    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p52    = m.add_instruction(pass_op{}, p51, p50);
    auto p53    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p54    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p55    = m.add_instruction(pass_op{}, p54, p52, p53);
    auto p56    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p61    = m.add_instruction(pass_op{}, p56, p55);
    auto p62    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p63    = m.add_instruction(pass_op{}, p62, p61, p41);
    auto p64    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p65    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p66    = m.add_instruction(pass_op{}, p65, p63, p64);
    auto p67    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p72    = m.add_instruction(pass_op{}, p67, p66);
    auto p73    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p74    = m.add_instruction(pass_op{}, p73, p72);
    auto p75    = add_alloc(m, {migraphx::shape::float_type, {0}});
    auto p76    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p77    = m.add_instruction(pass_op{}, p76, p74, p75);
    auto p78    = add_alloc(m, {migraphx::shape::float_type, {1, 64, 56, 56}});
    auto p83    = m.add_instruction(pass_op{}, p78, p77);
    m.add_instruction(pass_op{}, output, p83, p63);
    run_pass(m);
    CHECK(m.get_parameter_shape("scratch").bytes() == 7225344); // Optimal solution is 6422528
    CHECK(no_allocate(m));
}

TEST_CASE(test39)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond   = add_alloc(*mm, cond_s);
    auto output = mm->add_parameter("output", {migraphx::shape::float_type, {20}});

    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
    auto l1                  = mm->add_literal(migraphx::literal(ds, data1));
    std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
    auto l2                  = mm->add_literal(migraphx::literal(ds, data2));

    auto* then_mod = p.create_module("If_0_if");
    auto i1        = add_alloc(*then_mod, ds);
    auto a1        = then_mod->add_instruction(pass_op{}, i1, l1);
    then_mod->add_return({a1, output});

    auto* else_mod = p.create_module("If_0_else");
    auto i2        = add_alloc(*else_mod, ds);
    auto a2        = else_mod->add_instruction(pass_op{}, i2, l2);
    else_mod->add_return({a2, output});

    auto ret = mm->add_instruction(mod_pass_op{}, {cond}, {then_mod, else_mod});
    mm->add_return({ret, output});

    auto sub_modules = p.get_modules();
    std::reverse(sub_modules.begin(), sub_modules.end());
    for(auto& smod : sub_modules)
    {
        run_pass(*smod);
    }

    CHECK(mm->get_parameter_shape("scratch").bytes() == 4);
    CHECK(then_mod->get_parameter_shape("scratch").bytes() == 24);
    CHECK(else_mod->get_parameter_shape("scratch").bytes() == 24);
    CHECK(no_allocate(*mm));
    CHECK(no_allocate(*then_mod));
    CHECK(no_allocate(*else_mod));
}

TEST_CASE(literal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto lit = generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
    mm->add_literal(lit);
    run_pass(*mm);
    auto result = p.eval({}).back();
    CHECK(lit == result);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
