#include <migraphx/gpu/driver/perf.hpp>
#include <migraphx/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/time.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

std::vector<argument> generate_arguments(const std::vector<shape>& shapes, unsigned long seed = 0)
{
    std::vector<argument> args;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(args), [&](auto& s) {
        return to_gpu(generate_argument(s, seed++));
    });
    return args;
}

using milliseconds = std::chrono::duration<double, std::milli>;
double time_op(context& ctx, operation op, const std::vector<shape>& inputs, int n)
{
    // TODO: Use std::ref
    migraphx::context gctx = ctx;
    auto output            = op.compute_shape(inputs);
    op.finalize(gctx, output, inputs);
    auto args = generate_arguments(inputs);
    auto run  = [&] {
        op.compute(gctx, output, args);
        gctx.finish();
    };
    run();
    auto r   = range(n);
    double t = std::accumulate(
        r.begin(), r.end(), double{0.0}, [&](auto x, auto) { return x + time<milliseconds>(run); });
    return t / n;
}

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
