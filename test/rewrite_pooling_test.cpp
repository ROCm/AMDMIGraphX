#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/verify.hpp>

bool is_pooling(migraphx::instruction& ins) { return ins.name() == "pooling"; }
static void opt_pooling(migraphx::module& m)
{
    migraphx::rewrite_pooling rp;
    migraphx::dead_code_elimination dce;
    rp.apply(m);
    dce.apply(m);
}

TEST_CASE(rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&](const std::string& mode) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", mode},
                                                        {"padding", {0, 0, 0}},
                                                        {"stride", {1, 1, 1}},
                                                        {"lengths", {3, 4, 5}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    auto opt_program = [&](const migraphx::operation& reduce_op) {
        migraphx::module m;
        auto input = m.add_parameter("x", s);
        auto rsp   = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), input);
        auto rdm   = m.add_instruction(reduce_op, rsp);
        auto ret =
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 1, 1, 1}}}), rdm);
        m.add_return({ret});
        return m;
    };

    auto test_rewrite = [&](const std::string& mode, const migraphx::operation& op) {
        migraphx::module m1 = pooling_program(mode);
        migraphx::module m2 = opt_program(op);
        opt_pooling(m1);
        EXPECT(m1 == m2);
    };

    test_rewrite("average", migraphx::make_op("reduce_mean", {{"axes", {1}}}));
    test_rewrite("max", migraphx::make_op("reduce_max", {{"axes", {1}}}));
}

TEST_CASE(rewrite_avepooling_na1_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "average"},
                                                        {"padding", {0, 1, 0}},
                                                        {"stride", {1, 1, 1}},
                                                        {"lengths", {3, 4, 5}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_avepooling_na2_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "average"},
                                                        {"padding", {0, 0, 0}},
                                                        {"stride", {1, 2, 1}},
                                                        {"lengths", {3, 4, 5}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_avepooling_na3_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    auto pooling_program = [&]() {
        migraphx::module m;

        auto input = m.add_parameter("x", s);
        auto ret   = m.add_instruction(migraphx::make_op("pooling",
                                                       {{"mode", "max"},
                                                        {"padding", {0, 0, 0}},
                                                        {"stride", {1, 1, 1}},
                                                        {"lengths", {3, 3, 5}}}),
                                     input);
        m.add_return({ret});
        return m;
    };

    migraphx::module m1 = pooling_program();
    migraphx::module m2 = m1;

    opt_pooling(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(literal_rewrite_pooling_test)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3, 4, 5}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 1.0f);

    auto pooling_program = [&](const std::string& mode) {
        migraphx::program p;

        auto* mm   = p.get_main_module();
        auto input = mm->add_literal(migraphx::literal(s, data));
        auto ret   = mm->add_instruction(migraphx::make_op("pooling",
                                                         {{"mode", mode},
                                                          {"padding", {0, 0, 0}},
                                                          {"stride", {1, 1, 1}},
                                                          {"lengths", {3, 4, 5}}}),
                                       input);
        mm->add_return({ret});
        return p;
    };

    auto opt_program = [&](const migraphx::operation& op) {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto input = mm->add_literal(migraphx::literal(s, data));
        auto rsp   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, -1}}}), input);
        auto rdm   = mm->add_instruction(op, rsp);
        auto ret =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 1, 1, 1}}}), rdm);
        mm->add_return({ret});

        return p;
    };

    auto test_rewrite_pooling = [&](const std::string& mode, const migraphx::operation& op) {
        migraphx::program p1 = pooling_program(mode);
        migraphx::program p2 = opt_program(op);
        p1.compile(migraphx::ref::target{});
        p2.compile(migraphx::ref::target{});
        auto result1 = p1.eval({}).back();
        auto result2 = p2.eval({}).back();
        visit_all(result1,
                  result2)([&](auto r1, auto r2) { EXPECT(migraphx::verify_range(r1, r2)); });
    };

    test_rewrite_pooling("max", migraphx::make_op("reduce_max", {{"axes", {1}}}));
    test_rewrite_pooling("average", migraphx::make_op("reduce_mean", {{"axes", {1}}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
