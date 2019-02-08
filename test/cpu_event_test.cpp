#include <migraphx/pre_scheduling.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

// This is a test to trigger the code in cpu's context.hpp and runtime
// codes in program.cpp.
// 

TEST_CASE(test1)
{
    migraphx::program p;
    auto in1 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {32, 64, 1, 1}});
    auto in2 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}});
    auto p1  = p.add_instruction(migraphx::op::convolution{}, in1, in2);
    p1->set_stream(0);
    auto in3 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}});
    auto p2  = p.add_instruction(migraphx::op::convolution{}, in1, in3);
    p2->set_stream(1);
    p2->set_event(0);
    p2->add_mask(migraphx::record_event);
    auto p3 = p.add_instruction(migraphx::op::concat{1}, p1, p2);
    p3->set_stream(0);
    p3->add_mask(migraphx::wait_event);
    p.compile(migraphx::cpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::generate_argument(x.second);
    }
    p.eval(m);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
