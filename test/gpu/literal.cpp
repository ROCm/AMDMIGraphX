#include <test.hpp>
#include <basic_ops.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>

void gpu_literal_test()
{
    migraphx::program p;
    auto lit = generate_literal(migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
    p.add_literal(lit);
    p.compile(migraphx::gpu::target{});
    auto scratch = p.get_parameter("scratch");
    if(scratch == p.end())
    {
        auto result = p.eval({}).back();
        EXPECT(lit == migraphx::gpu::from_gpu(result));
    }
    else
    {
        EXPECT(scratch->get_shape().bytes() == lit.get_shape().bytes());
    }
}

int main() { gpu_literal_test(); }
