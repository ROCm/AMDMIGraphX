#include <test.hpp>
#include <basic_ops.hpp>
#include <migraph/program.hpp>
#include <migraph/generate.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/hip.hpp>

void gpu_literal_test()
{
    migraph::program p;
    auto lit = generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
    p.add_literal(lit);
    p.compile(migraph::gpu::target{});
    auto result = p.eval({});
    EXPECT(lit == migraph::gpu::from_gpu(result));
}

int main() { gpu_literal_test(); }
