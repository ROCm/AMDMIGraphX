#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"
#include <migraphx/half.hpp>

TEST_CASE(param_add)
{
    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1 = p.add_parameter("x", s);
        auto p2 = p.add_parameter("y", s);
        p.add_instruction(migraphx::op::add{}, p1, p2);

        return p;
    };

    auto create_program_half = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto hp1 = p.insert_instruction(std::next(p1), migraphx::op::convert{}, p1);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(std::next(p2), migraphx::op::convert{}, p2);
        auto hs  = p.add_instruction(migraphx::op::add{}, hp1, hp2);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hs);

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize(p1);
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize(p1, {"add"});
        EXPECT(p1 == p2);
    }
}

TEST_CASE(param_add_sub)
{
    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1   = p.add_parameter("x", s);
        auto p2   = p.add_parameter("y", s);
        auto sum  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto diff = p.add_instruction(migraphx::op::sub{}, sum, p2);
        p.add_instruction(migraphx::op::add{}, diff, p1);

        return p;
    };

    auto create_program_half_add = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto hp1 = p.insert_instruction(
            std::next(p1), migraphx::op::convert{migraphx::shape::half_type}, p1);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(
            std::next(p2), migraphx::op::convert{migraphx::shape::half_type}, p2);
        auto hsum  = p.add_instruction(migraphx::op::add{}, hp1, hp2);
        auto sum   = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hsum);
        auto diff  = p.add_instruction(migraphx::op::sub{}, sum, p2);
        auto hdiff = p.add_instruction(
            migraphx::op::convert{migraphx::op::convert{migraphx::shape::half_type}}, diff);
        auto res = p.add_instruction(migraphx::op::add{}, hdiff, hp1);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, res);

        return p;
    };

    auto create_program_half_sub = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(
            std::next(p2), migraphx::op::convert{migraphx::shape::half_type}, p2);
        auto sum   = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto hsum  = p.add_instruction(migraphx::op::convert{migraphx::shape::half_type}, sum);
        auto hdiff = p.add_instruction(migraphx::op::sub{}, hsum, hp2);
        auto diff  = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hdiff);
        p.add_instruction(migraphx::op::add{}, diff, p1);

        return p;
    };

    auto create_program_half_all = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto p1  = p.add_parameter("x", s);
        auto hp1 = p.insert_instruction(
            std::next(p1), migraphx::op::convert{migraphx::shape::half_type}, p1);
        auto p2  = p.add_parameter("y", s);
        auto hp2 = p.insert_instruction(
            std::next(p2), migraphx::op::convert{migraphx::shape::half_type}, p2);
        auto hsum  = p.add_instruction(migraphx::op::add{}, hp1, hp2);
        auto hdiff = p.add_instruction(migraphx::op::sub{}, hsum, hp2);
        auto hres  = p.add_instruction(migraphx::op::add{}, hdiff, hp1);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hres);

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_add();

        migraphx::quantize(p1, {"add"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_sub();

        migraphx::quantize(p1, {"sub"});
        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half_all();

        migraphx::quantize(p1);
        migraphx::run_passes(p1, {migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }
}

TEST_CASE(literal_add)
{
    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        std::vector<float> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = p.add_literal(migraphx::literal(s, data));
        auto l2 = p.add_literal(migraphx::literal(s, data));
        p.add_instruction(migraphx::op::add{}, l1, l2);

        return p;
    };

    auto create_program_half = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::half_type, {2, 3}};
        std::vector<migraphx::half> data(2 * 3);
        std::iota(data.begin(), data.end(), 1.0f);
        auto l1 = p.add_literal(migraphx::literal(s, data));
        auto l2 = p.add_literal(migraphx::literal(s, data));
        auto hs = p.add_instruction(migraphx::op::add{}, l1, l2);
        p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, hs);

        return p;
    };

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize(p1, {"all"});
        migraphx::run_passes(p1,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(p2,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});

        EXPECT(p1 == p2);
    }

    {
        auto p1 = create_program_float();
        auto p2 = create_program_half();

        migraphx::quantize(p1, {"add"});
        migraphx::run_passes(p1,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        migraphx::run_passes(p2,
                             {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
        EXPECT(p1 == p2);
    }
}

TEST_CASE(op_capture)
{
    auto test_func = [&](std::size_t ins_index, std::vector<migraphx::argument> args) {
        (void)ins_index;
        (void)args;
    };

    auto create_program_float = [] {
        migraphx::program p;
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1 = p.add_parameter("x", s1);
        auto p2 = p.add_parameter("y", s1);
        auto pb = p.add_parameter("b", s2);
        auto pc = p.add_parameter("c", s2);
        auto pa = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto ps = p.add_instruction(migraphx::op::dot{}, pa, pb, pc);
        p.add_instruction(migraphx::op::dot{}, pa, ps);

        return p;
    };

    auto create_program_op = [&] {
        migraphx::program p;
        migraphx::shape s1{migraphx::shape::float_type, {3, 3}};
        migraphx::shape s2{migraphx::shape::float_type, {3, 6}};

        auto p1  = p.add_parameter("x", s1);
        auto p2  = p.add_parameter("y", s1);
        auto pb  = p.add_parameter("b", s2);
        auto pc  = p.add_parameter("c", s2);
        auto pa  = p.add_instruction(migraphx::op::add{}, p1, p2);
        auto opb = p.insert_instruction(std::next(pb), migraphx::op::capture{1, test_func}, pb);
        auto opc = p.insert_instruction(std::next(pc), migraphx::op::capture{2, test_func}, pc);
        auto opa = p.add_instruction(migraphx::op::capture{0, test_func}, pa);
        auto ps  = p.add_instruction(migraphx::op::dot{}, opa, opb, opc);
        auto ops = p.add_instruction(migraphx::op::capture{3, test_func}, ps);
        p.add_instruction(migraphx::op::dot{}, opa, ops);

        return p;
    };

    {
        auto p            = create_program_float();
        auto op_capture_p = create_program_op();
        migraphx::capture_arguments(p);
        EXPECT(p == op_capture_p);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
