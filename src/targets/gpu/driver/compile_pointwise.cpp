#include <migraphx/gpu/driver/action.hpp>
#include <migraphx/gpu/driver/perf.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

struct compile_pointwise : action<compile_pointwise>
{
    static void apply(const parser& p, const value& v)
    {
        context ctx;
        auto inputs = p.parse_shapes(v.at("inputs"));
        // inputs is a vector of shape
for( auto sh : inputs)
{
    auto mystride = sh.strides();
    printf("brian 1 says the stride is %lu\n", mystride[0]);
}

        int global_workitems = p.get(v, "global", 64);
        int local_workitems_per_CU = p.get(v, "local", 64);

        auto op     = gpu::compile_pointwise(ctx, inputs, v.at("lambda").to<std::string>(), "", global_workitems, local_workitems_per_CU );

        double t    = time_op(ctx, op, inputs, p.get(v, "iterations", 100));
        std::cout << op << ": " << t << "ms" << std::endl;
    }
};

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
