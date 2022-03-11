#ifndef MIGRAPHX_GUARD_GPU_COMPILER_HPP
#define MIGRAPHX_GUARD_GPU_COMPILER_HPP

#include <migraphx/config.hpp>
#include <migraphx/auto_register.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

using compiler_replace = std::function<void(module& m, instruction_ref ins)>;
using compiler_compile = std::function<compiler_replace(context&, instruction_ref, operation)>;
using compiler_compile_op =
    std::function<operation(context&, const std::vector<shape>& inputs, const value&)>;

void register_compiler(const std::string& name, compiler_compile c, compiler_compile_op cop);

bool has_compiler_for(const std::string& name);
compiler_replace compile(context& ctx, instruction_ref ins, const operation& op);
operation
compile_op(const std::string& name, context& ctx, const std::vector<shape>& inputs, const value& v);

template <class T>
void register_compiler()
{
    T c;
    for(auto&& name : c.names())
    {
        register_compiler(
            name,
            [=](auto&&... xs) { return c.compile(std::forward<decltype(xs)>(xs)...); },
            [=](auto&&... xs) { return c.compile_op(std::forward<decltype(xs)>(xs)...); });
    }
}

struct register_compiler_action
{
    template <class T>
    static void apply()
    {
        register_compiler<T>();
    }
};

template <class T>
using auto_register_compiler = auto_register<register_compiler_action, T>;

template <class Derived>
struct compiler : auto_register_compiler<Derived>
{
    auto replace(const operation& op) const
    {
        return
            [=](module& m, instruction_ref ins) { m.replace_instruction(ins, op, ins->inputs()); };
    }
    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_COMPILER_HPP
