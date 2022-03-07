#include <migraphx/gpu/driver/action.hpp>
#include <migraphx/gpu/driver/perf.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace driver {

struct compile_op : action<compile_op>
{
    static void apply(const parser& p, const value& v)
    {
        context ctx;
        auto inputs = p.parse_shapes(v.at("inputs"));
        auto op = gpu::compile_op(v.at("name").to<std::string>(), ctx, inputs, v);
        double t    = time_op(ctx, op, inputs, p.get(v, "iterations", 100));
        std::cout << op << ": " << t << "ms" << std::endl;
    }
};

} // namespace driver
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
