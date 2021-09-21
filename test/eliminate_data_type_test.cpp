#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m, std::set<migraphx::shape::type_t> types)
{
    migraphx::run_passes(
        m,
        {migraphx::eliminate_data_type{std::move(types), migraphx::shape::float_type},
         migraphx::eliminate_identity{},
         migraphx::dead_code_elimination{}});
}

TEST_CASE(simple)
{
    migraphx::shape s{migraphx::shape::int8_type, {2, 2}};
    migraphx::module mm1;
    {
        auto x = mm1.add_parameter("x", s);
        auto y = mm1.add_parameter("y", s);
        mm1.add_instruction(migraphx::make_op("add"), x, y);
    }
    run_pass(mm1, {migraphx::shape::int8_type});

    migraphx::module mm2;
    {
        auto x      = mm2.add_parameter("x", s);
        auto y      = mm2.add_parameter("y", s);
        auto floatx = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto floaty = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto add = mm2.add_instruction(migraphx::make_op("add"), floatx, floaty);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}), add);
    }
    EXPECT(mm1 == mm2);
}

TEST_CASE(quant)
{
    migraphx::shape s{migraphx::shape::int8_type, {2, 2}};
    migraphx::module mm1;
    {
        auto x = mm1.add_parameter("x", s);
        auto y = mm1.add_parameter("y", s);
        mm1.add_instruction(migraphx::make_op("quant_dot"), x, y);
    }
    run_pass(mm1, {migraphx::shape::int8_type});

    migraphx::module mm2;
    {
        auto x      = mm2.add_parameter("x", s);
        auto y      = mm2.add_parameter("y", s);
        auto floatx = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto floaty = mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto add = mm2.add_instruction(migraphx::make_op("dot"), floatx, floaty);
        mm2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), add);
    }
    EXPECT(mm1 == mm2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
