#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct pointwise_compiler
{
    std::string name() const
    {
        return "pointwise";
    }

    operation apply(context& ctx, instruction_ref ins) const
    {
        assert(not ins->module_inputs().empty());
        auto* pm = ins->module_inputs().front();
        auto inputs = to_shapes(ins->inputs());
        inputs.push_back(ins->get_shape());
        return compile_pointwise(ctx, inputs, *pm);
    }
};

using compiler_function = std::function<operation(context&, instruction_ref)>;

template<class T>
compiler_function make_compiler_function(T x)
{
    return {[=](context& ctx, instruction_ref ins) {
        return x.apply(ctx, ins);
    }};
}

template<class... Ts>
std::unordered_map<std::string, compiler_function> make_compilers(Ts... xs)
{
    return {
        {xs.name(), make_compiler_function(xs)}...
    };
}

void compile_ops::apply(module& m) const
{
    auto compilers = make_compilers(pointwise_compiler{});
    auto insert_allocation = alloc.allocation_inserter(m);
    for(auto ins:iterator_for(m))
    {
        if (not contains(compilers, ins->name()))
            continue;
        auto op = compilers[ins->name()](*ctx, ins);
        auto inputs = ins->inputs();
        inputs.push_back(insert_allocation(ins, ins->get_shape()));
        m.replace_instruction(ins, op, inputs);
    }
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
