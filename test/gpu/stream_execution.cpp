#include <test.hpp>
#include <basic_ops.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/verify_args.hpp>

migraphx::program create_program()
{
    migraphx::program p;
    auto in1 = p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {32, 64, 1, 1}});
    auto in2 = p.add_parameter("1", migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}});
    auto p1  = p.add_instruction(migraphx::op::convolution{}, in1, in2);
    auto in3 = p.add_parameter("2", migraphx::shape{migraphx::shape::float_type, {64, 64, 1, 1}});
    auto p2  = p.add_instruction(migraphx::op::convolution{}, in1, in3);
    p.add_instruction(migraphx::op::concat{1}, p1, p2);
    return p;
}

migraphx::argument run_gpu()
{
    setenv("MIGRAPHX_DISABLE_NULL_STREAM", "1", 1);
    migraphx::program p = create_program();
    p.compile(migraphx::gpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
    }
    auto ret_val = migraphx::gpu::from_gpu(p.eval(m));
    p.finish();
    return ret_val;
}

migraphx::argument run_cpu()
{
    migraphx::program p = create_program();
    p.compile(migraphx::cpu::target{});
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::generate_argument(x.second);
    }
    return p.eval(m);
}

void gpu_stream_execution_test()
{
    auto result1 = run_gpu();
    auto result2 = run_cpu();
    verify_args("test", result2, result1);
}

int main() { gpu_stream_execution_test(); }
