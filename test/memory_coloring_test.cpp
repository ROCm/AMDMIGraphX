#include <migraph/memory_coloring.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct memory_coloring_target
{
    std::string name() const { return "memory_coloring"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::memory_coloring{"allocate", true}};
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

migraph::instruction_ref add_alloc(migraph::program& p, const migraph::shape& s)
{
    auto a0 = p.add_outline(s);
    return p.add_instruction(allocate{}, a0);
}

bool no_allocate(const migraph::program& p)
{
    return std::none_of(p.begin(), p.end(), [](auto&& ins) { return ins.name() == "allocate"; });
}


void test1()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test2()
{
    migraph::program p;
    auto input = p.add_parameter("input", migraph::shape{migraph::shape::float_type, {16}});

    auto a1 = add_alloc(p, {migraph::shape::float_type, {128}});
    auto p1 = p.add_instruction(pass_op{}, a1, input);
    auto p2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(p));
}

void test3()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p2 = add_alloc(p, {migraph::shape::float_type, {128}});
    auto p1 = p.add_instruction(pass_op{}, p2, a1);
    auto p3 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p3, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(p));
}

void test4()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {0}});
    auto p2 = add_alloc(p, {migraph::shape::float_type, {128}});
    auto p1 = p.add_instruction(pass_op{}, p2, a1);
    auto p3 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p3, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 672);
    CHECK(no_allocate(p));
}

void test5()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test6()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p3 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

void test7()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p3 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

void test8()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p3 = add_alloc(p, {migraph::shape::float_type, {192}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 960);
    CHECK(no_allocate(p));
}

void test9()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p3 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, p3, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 96);
    CHECK(no_allocate(p));
}

void test10()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 32);
    CHECK(no_allocate(p));
}

void test11()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

void test12()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

void test13()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

void test14()
{
    migraph::program p;
    auto a3 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, a3, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 224);
    CHECK(no_allocate(p));
}

void test15()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a3, p1, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

void test16()
{
    migraph::program p;
    auto a1 = p.add_literal(migraph::generate_literal({migraph::shape::float_type, {8}}));
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_literal(migraph::generate_literal({migraph::shape::float_type, {40}}));
    auto p2 = p.add_instruction(pass_op{}, a2);
    auto a3 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a3, p1, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(p));
}

void test17()
{
    migraph::program p;
    auto a3 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto a1 = p.add_literal(migraph::generate_literal({migraph::shape::float_type, {8}}));
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_literal(migraph::generate_literal({migraph::shape::float_type, {40}}));
    auto p2 = p.add_instruction(pass_op{}, a2);
    p.add_instruction(pass_op{}, a3, p1, p2);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 160);
    CHECK(no_allocate(p));
}

void test18()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto p2 = p.add_instruction(pass_op{}, a1, p1);
    auto p3 = p.add_instruction(pass_op{}, p2, p1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1, p2, p3);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test19()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    auto a3 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a3, p2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 352);
    CHECK(no_allocate(p));
}

void test20()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraph::shape::float_type, {32}});
    p.add_instruction(pass_op{}, a4, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(p));
}

void test21()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(p));
}

void test22()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(p));
}

void test23()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 288);
    CHECK(no_allocate(p));
}

void test24()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto a3 = add_alloc(p, {migraph::shape::float_type, {32}});
    auto p1 = p.add_instruction(pass_op{}, a1, a2, a3);
    auto a4 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(pass_op{}, a4, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 384);
    CHECK(no_allocate(p));
}

void test25()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(nop{});
    auto p1 = p.add_instruction(pass_op{}, a1);
    p.add_instruction(nop{});
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test26()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    p.add_instruction(nop{}, a1);
    auto p1 = p.add_instruction(pass_op{}, a1);
    p.add_instruction(nop{}, a1, p1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(pass_op{}, a2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test27()
{
    migraph::program p;
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.add_instruction(nop{}, a2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test28()
{
    migraph::program p;
    auto output = p.add_parameter("output", {migraph::shape::float_type, {8}});
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.add_instruction(pass_op{}, p2, output);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test29()
{
    migraph::program p;
    auto output = p.add_parameter("output", {migraph::shape::float_type, {8}});
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.move_instruction(output, p2);
    p.add_instruction(pass_op{}, p2, output);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test30()
{
    migraph::program p;
    auto output = p.add_parameter("x", {migraph::shape::float_type, {8}});
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);
    p.move_instruction(output, p2);
    p.add_instruction(pass_op{}, p2, output);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void test31()
{
    migraph::program p;
    auto output = p.add_parameter("output", {migraph::shape::float_type, {8}});
    auto a1 = add_alloc(p, {migraph::shape::float_type, {8}});
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = add_alloc(p, {migraph::shape::float_type, {40}});
    p.move_instruction(output, a2);
    p.add_instruction(pass_op{}, a2, p1);
    p.compile(memory_coloring_target{});
    CHECK(p.get_parameter_shape("scratch").bytes() == 192);
    CHECK(no_allocate(p));
}

void literal_test()
{
    migraph::program p;
    auto lit = generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
    p.add_literal(lit);
    p.compile(memory_coloring_target{});
    auto result = p.eval({});
    CHECK(lit == result);
}

int main()
{
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();
    test11();
    test12();
    test13();
    test14();
    test15();
    test16();
    test17();
    // test18();
    test19();
    test20();
    test21();
    test22();
    test23();
    test24();
    test25();
    // test26();
    test27();
    test28();
    test29();
    test30();
    test31();

    literal_test();
}
