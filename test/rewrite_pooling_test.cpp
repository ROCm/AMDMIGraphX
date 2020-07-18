#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/verify.hpp>

bool is_pooling(migraphx::instruction& ins) { return ins.name() == "pooling"; }
static void opt_pooling(migraphx::program& prog)
{
    migraphx::rewrite_pooling rp;
    migraphx::dead_code_elimination dce;
    rp.apply(prog);
    dce.apply(prog);
}

TEST_CASE(rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&](const std::string& mode) {
        migraphx::program p;
        auto input = p.add_parameter("x", s);
        auto ret =
            p.add_instruction(migraphx::op::pooling{mode, {0, 0, 0}, {1, 1, 1}, {3, 4, 5}}, input);
        p.add_return({ret});
        return p;
    };

    auto opt_program = [&](const migraphx::operation& reduce_op) {
        migraphx::program p;
        auto input = p.add_parameter("x", s);
        auto rsp   = p.add_instruction(migraphx::op::reshape{{4, -1}}, input);
        auto rdm   = p.add_instruction(reduce_op, rsp);
        auto ret   = p.add_instruction(migraphx::op::reshape{{2, 2, 1, 1, 1}}, rdm);
        p.add_return({ret});
        return p;
    };

    auto test_rewrite = [&](const std::string& mode, const migraphx::operation& op) {
        migraphx::program p1 = pooling_program(mode);
        migraphx::program p2 = opt_program(op);
        opt_pooling(p1);
        EXPECT(p1 == p2);
    };

    test_rewrite("average", migraphx::op::reduce_mean{{1}});
    test_rewrite("max", migraphx::op::reduce_max{{1}});
}

TEST_CASE(rewrite_avepooling_na1_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::program p;
        auto input = p.add_parameter("x", s);
        auto ret   = p.add_instruction(
            migraphx::op::pooling{"average", {0, 1, 0}, {1, 1, 1}, {3, 4, 5}}, input);
        p.add_return({ret});
        return p;
    };

    migraphx::program p1 = pooling_program();
    migraphx::program p2 = p1;
    opt_pooling(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(rewrite_avepooling_na2_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::program p;
        auto input = p.add_parameter("x", s);
        auto ret   = p.add_instruction(
            migraphx::op::pooling{"average", {0, 0, 0}, {1, 2, 1}, {3, 4, 5}}, input);
        p.add_return({ret});
        return p;
    };

    migraphx::program p1 = pooling_program();
    migraphx::program p2 = p1;
    opt_pooling(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(rewrite_avepooling_na3_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::program p;
        auto input = p.add_parameter("x", s);
        auto ret =
            p.add_instruction(migraphx::op::pooling{"max", {0, 0, 0}, {1, 1, 1}, {3, 3, 5}}, input);
        p.add_return({ret});
        return p;
    };

    migraphx::program p1 = pooling_program();
    migraphx::program p2 = p1;
    opt_pooling(p1);
    EXPECT(p1 == p2);
}

TEST_CASE(literal_rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 1.0f);

    auto pooling_program = [&](const std::string& mode) {
        migraphx::program p;
        auto input = p.add_literal(migraphx::literal(s, data));
        auto ret =
            p.add_instruction(migraphx::op::pooling{mode, {0, 0, 0}, {1, 1, 1}, {3, 4, 5}}, input);
        p.add_return({ret});
        return p;
    };

    auto opt_program = [&](const migraphx::operation& op) {
        migraphx::program p;
        auto input = p.add_literal(migraphx::literal(s, data));
        auto rsp   = p.add_instruction(migraphx::op::reshape{{4, -1}}, input);
        auto rdm   = p.add_instruction(op, rsp);
        auto ret   = p.add_instruction(migraphx::op::reshape{{2, 2, 1, 1, 1}}, rdm);
        p.add_return({ret});

        return p;
    };

    auto test_rewrite_pooling = [&](const std::string& mode, const migraphx::operation& op) {
        migraphx::program p1 = pooling_program(mode);
        migraphx::program p2 = opt_program(op);
        p1.compile(migraphx::cpu::target{});
        p2.compile(migraphx::cpu::target{});
        auto result1 = p1.eval({}).back();
        auto result2 = p2.eval({}).back();
        visit_all(result1,
                  result2)([&](auto r1, auto r2) { EXPECT(migraphx::verify_range(r1, r2)); });
    };

    test_rewrite_pooling("max", migraphx::op::reduce_max{{1}});
    test_rewrite_pooling("average", migraphx::op::reduce_mean{{1}});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
