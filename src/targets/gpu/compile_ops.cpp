#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct precompile_op
{
    operation op = op::identity{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::precompile_op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        inputs.pop_back();
        return op.compute_shape(inputs, mods);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

MIGRAPHX_REGISTER_OP(precompile_op);

struct pointwise_compiler
{
    std::string name() const { return "pointwise"; }

    operation apply(context& ctx, instruction_ref ins, const operation&) const
    {
        assert(not ins->module_inputs().empty());
        auto* pm = ins->module_inputs().front();
        return compile_pointwise(ctx, to_shapes(ins->inputs()), *pm);
    }
};

using compiler_function = std::function<operation(context&, instruction_ref, operation)>;

template <class T>
compiler_function make_compiler_function(T x)
{
    return {[=](auto&&... xs) { return x.apply(xs...); }};
}

template <class... Ts>
std::unordered_map<std::string, compiler_function> make_compilers(Ts... xs)
{
    return {{xs.name(), make_compiler_function(xs)}...};
}

void compile_ops::apply(module& m) const
{
    auto compilers = make_compilers(pointwise_compiler{});
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::precompile_op")
            continue;
        operation preop = any_cast<precompile_op>(ins->get_operator()).op;
        assert(contains(compilers, preop.name()));
        auto op = compilers[preop.name()](*ctx, ins, preop);
        m.replace_instruction(ins, op, ins->inputs());
    }
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
