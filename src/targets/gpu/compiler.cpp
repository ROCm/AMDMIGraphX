#include <migraphx/gpu/compiler.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

auto& compiler_map()
{
    static std::unordered_map<std::string, compiler_compile> m; // NOLINT
    return m;
}

auto& compiler_op_map()
{
    static std::unordered_map<std::string, compiler_compile_op> m; // NOLINT
    return m;
}

void register_compiler(const std::string& name, compiler_compile c, compiler_compile_op cop)
{
    compiler_map()[name] = c;
    compiler_op_map()[name] = cop;
}

bool has_compiler_for(const std::string& name)
{
    return compiler_map().count(name) > 0;
}
compiler_replace compile(context& ctx, instruction_ref ins, operation op)
{
    return compiler_map().at(op.name())(ctx, ins, op);
}
operation compile_op(const std::string& name, context& ctx, const std::vector<shape>& inputs, const value& v)
{
    return compiler_op_map().at(name)(ctx, inputs, v);
}


} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

