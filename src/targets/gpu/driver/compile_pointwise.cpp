#include <migraphx/gpu/driver/action.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>
#include <migraphx/gpu/context.hpp>

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
double time_op(context& ctx, operation op, const std::vector<shape>& inputs)
{
    migraphx::context gctx = ctx;
    auto output            = op.compute_shape(inputs);
    op.finalize(gctx, output, inputs);
    auto args = generate_arguments(inputs);
    return time<milliseconds>([&] {
        op.compute(gctx, output, args);
        gctx.finish();
    });
}

struct compile_pointwise : action<compile_pointwise>
{
    static void apply(const parser& p, const value& v)
    {
        context ctx;
        auto inputs = p.parse_shapes(v.at("inputs"));
        auto op     = gpu::compile_pointwise(ctx, inputs, v.at("lambda").to<std::string>());
        double t    = time_op(ctx, op, inputs);
        std::cout << op << ": " << t << "ms" << std::endl;
    }
};

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
